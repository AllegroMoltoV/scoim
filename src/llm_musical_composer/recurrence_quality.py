"""多尺度再現の校正に使う声部アンカーと区分呼吸を診断する。"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

from llm_musical_composer.performance_pipeline import (
    PiecePlan,
    PlanNode,
    RenderedPerformance,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    harmony_pitch_classes,
)


class RecurrenceQualityError(ValueError):
    """校正用の品質診断を実行できない入力を表す。"""


@dataclass(frozen=True)
class VoiceTextureAssessment:
    material_id: str
    opening_anchor: int | None
    closing_anchor: int | None
    internal_independence: bool
    passes: bool


@dataclass(frozen=True)
class BoundaryBreathAssessment:
    node_id: str
    new_attack_gap_units: int
    minimum_gap_units: int
    passes: bool


@dataclass(frozen=True)
class PianoTextureAssessment:
    material_id: str
    attack_size_counts: tuple[tuple[int, int], ...]
    maximum_polyphony: int
    passes: bool


@dataclass(frozen=True)
class SectionPacingAssessment:
    source_node_id: str
    target_node_id: str
    source_ms_per_unit: float
    target_ms_per_unit: float
    source_note_count: int
    target_note_count: int
    maximum_target_ratio: float
    passes: bool


@dataclass(frozen=True)
class LocalPulseAssessment:
    node_id: str
    voice: str
    interval_count: int
    minimum_ms_per_unit: float
    median_ms_per_unit: float
    maximum_ms_per_unit: float
    variation_ratio: float
    maximum_adjacent_ratio: float
    simultaneous_attack_max_spread_ms: int
    minimum_variation_ratio: float
    maximum_allowed_adjacent_ratio: float
    passes: bool


@dataclass(frozen=True)
class SectionContrastAssessment:
    target_node_id: str
    source_node_id: str
    changed_axes: tuple[str, ...]
    rhythm_grid_distance: float
    articulation_ratio_delta: float
    shared_pitch_class_ratio: float
    passes: bool


@dataclass(frozen=True)
class VerticalAlignmentAssessment:
    source_node_id: str
    target_node_id: str
    source_ratio: float
    target_ratio: float
    minimum_target_ratio: float
    minimum_increase: float
    passes: bool


@dataclass(frozen=True)
class MaterialAlignmentAssessment:
    material_id: str
    shared_attack_ratio: float
    internal_independence: bool
    minimum_ratio: float
    passes: bool


@dataclass(frozen=True)
class MaterialHarmonyAssessment:
    material_id: str
    status: str
    accompaniment_voice: str | None
    accompaniment_solo_attacks: int
    accompaniment_chord_attacks: int
    arpeggio_candidates: int
    accompaniment_chord_tone_ratio: float | None
    low_spacing_violations: int
    foreground_non_chord_tones: int
    resolved_foreground_non_chord_tones: int
    unresolved_foreground_non_chord_tones: int
    passes: bool


@dataclass(frozen=True)
class RenderedHarmonyAssessment:
    status: str
    pedal_carryover_violations: int
    passes: bool


@dataclass(frozen=True)
class ForegroundVariationAssessment:
    source_material_id: str
    target_material_id: str
    voice: str
    source_note_count: int
    target_note_count: int
    pitch_difference_count: int
    rhythm_difference_count: int
    articulation_difference_count: int
    foreground_motif_head_preserved: bool
    exact_surface_copy: bool
    passes: bool


@dataclass(frozen=True)
class TransitionConnectionAssessment:
    source_node_id: str
    transition_node_id: str
    target_node_id: str
    voice: str
    transition_attack_count: int
    source_to_transition_semitones: int
    maximum_internal_leap: int
    transition_to_target_semitones: int
    same_pitch_restrike: bool
    passes: bool


@dataclass(frozen=True)
class RenderedBoundaryAssessment:
    transition_node_id: str
    target_node_id: str
    voice: str
    transition_last_attack_ms: int
    target_first_attack_ms: int
    foreground_note_gap_ms: int
    same_pitch_restrike: bool
    pedal_gap_ms: int
    passes: bool


@dataclass(frozen=True)
class ForegroundDissonanceTone:
    material_id: str
    event_id: str
    resolution_event_id: str | None
    harmony_id: str | None
    role: str
    approach_semitones: int | None
    departure_semitones: int | None


@dataclass(frozen=True)
class ForegroundDissonanceAssessment:
    material_id: str
    status: str
    passing_tones: int
    neighbor_tones: int
    unsupported_tones: int
    tones: tuple[ForegroundDissonanceTone, ...]
    passes: bool


@dataclass(frozen=True)
class RenderedDissonanceOverlap:
    occurrence_node_id: str
    event_id: str
    resolution_event_id: str
    resolution_at_ms: int
    sounding_until_ms: int
    overlap_ms: int


@dataclass(frozen=True)
class RenderedForegroundDissonanceAssessment:
    status: str
    pedal_overlap_count: int
    maximum_overlap_ms: int
    overlaps: tuple[RenderedDissonanceOverlap, ...]


def _chord_pitch_classes(root: int, quality: str) -> frozenset[int]:
    return harmony_pitch_classes(root, quality)


def analyze_foreground_dissonance(
    score: ScoreSpec,
    material_id: str,
) -> ForegroundDissonanceAssessment:
    """前景和声外音を前後3発音から経過音、刺繍音、条件外へ分類する。"""
    materials = {material.material_id: material for material in score.materials}
    if material_id not in materials:
        raise RecurrenceQualityError("dissonance material does not exist")
    material = materials[material_id]
    if not material.harmonies or material.foreground_voice is None:
        return ForegroundDissonanceAssessment(material_id, "unassessed", 0, 0, 0, (), False)
    foreground = tuple(note for note in material.notes if note.voice == material.foreground_voice)
    by_onset: dict[int, list[ScoreNote]] = defaultdict(list)
    for note in foreground:
        by_onset[note.at_units].append(note)
    select = max if material.foreground_voice == "upper" else min
    representatives = tuple(
        select(by_onset[onset], key=lambda note: note.pitch) for onset in sorted(by_onset)
    )

    def harmony_at(note: ScoreNote):
        return next(
            (
                harmony
                for harmony in material.harmonies
                if harmony.at_units <= note.at_units < harmony.at_units + harmony.duration_units
            ),
            None,
        )

    tones: list[ForegroundDissonanceTone] = []
    for index, note in enumerate(representatives):
        harmony = harmony_at(note)
        if harmony is not None and note.pitch % 12 in _chord_pitch_classes(
            harmony.root_pitch_class,
            harmony.quality,
        ):
            continue
        previous = representatives[index - 1] if index else None
        following = representatives[index + 1] if index + 1 < len(representatives) else None
        previous_harmony = harmony_at(previous) if previous is not None else None
        following_harmony = harmony_at(following) if following is not None else None
        approach = note.pitch - previous.pitch if previous is not None else None
        departure = following.pitch - note.pitch if following is not None else None
        common_context = (
            harmony is not None
            and previous is not None
            and following is not None
            and previous_harmony is not None
            and following_harmony is not None
            and previous_harmony.harmony_id == harmony.harmony_id == following_harmony.harmony_id
            and previous.pitch % 12
            in _chord_pitch_classes(harmony.root_pitch_class, harmony.quality)
            and following.pitch % 12
            in _chord_pitch_classes(harmony.root_pitch_class, harmony.quality)
            and note.at_units != harmony.at_units
            and note.duration_units * 2 <= score.divisions
            and note.at_units + note.duration_units <= harmony.at_units + harmony.duration_units
        )
        role = "unsupported"
        if (
            common_context
            and approach is not None
            and departure is not None
            and abs(approach) in {1, 2}
            and abs(departure) in {1, 2}
        ):
            if approach * departure > 0:
                role = "passing"
            elif approach * departure < 0 and previous.pitch == following.pitch:
                role = "neighbor"
        tones.append(
            ForegroundDissonanceTone(
                material.material_id,
                note.event_id,
                following.event_id if following is not None else None,
                harmony.harmony_id if harmony is not None else None,
                role,
                approach,
                departure,
            )
        )
    result = tuple(tones)
    passing = sum(tone.role == "passing" for tone in result)
    neighbor = sum(tone.role == "neighbor" for tone in result)
    unsupported = sum(tone.role == "unsupported" for tone in result)
    return ForegroundDissonanceAssessment(
        material.material_id,
        "assessed",
        passing,
        neighbor,
        unsupported,
        result,
        unsupported == 0,
    )


def analyze_rendered_foreground_dissonance(
    plan: PiecePlan,
    score: ScoreSpec,
    rendered: RenderedPerformance,
) -> RenderedForegroundDissonanceAssessment:
    """機能的な前景非和声音が解決時にも鳴るペダル保持を返す。"""
    materials = {material.material_id: material for material in score.materials}
    performed = {(note.occurrence_node_id, note.event_id): note for note in rendered.notes}
    pedals = tuple(sorted(rendered.pedals, key=lambda item: (item.at_ms, item.value)))

    def pedal_down_at(at_ms: int) -> bool:
        state = 0
        for pedal in pedals:
            if pedal.at_ms > at_ms:
                break
            state = pedal.value
        return state >= 64

    def sounding_until(note) -> int:
        note_off = note.at_ms + note.duration_ms
        if not pedal_down_at(note_off):
            return note_off
        return next(
            (pedal.at_ms for pedal in pedals if pedal.at_ms >= note_off and pedal.value < 64),
            note_off,
        )

    overlaps: list[RenderedDissonanceOverlap] = []
    assessed = False
    for node in plan.nodes:
        if node.score_material_id is None:
            continue
        material = materials[node.score_material_id]
        assessment = analyze_foreground_dissonance(score, material.material_id)
        if assessment.status != "assessed":
            continue
        assessed = True
        for tone in assessment.tones:
            if tone.role not in {"passing", "neighbor"} or tone.resolution_event_id is None:
                continue
            event_id = f"{node.node_id}:{tone.event_id}"
            resolution_event_id = f"{node.node_id}:{tone.resolution_event_id}"
            note = performed[(node.node_id, event_id)]
            resolution = performed[(node.node_id, resolution_event_id)]
            release = sounding_until(note)
            if release <= resolution.at_ms:
                continue
            overlaps.append(
                RenderedDissonanceOverlap(
                    node.node_id,
                    tone.event_id,
                    tone.resolution_event_id,
                    resolution.at_ms,
                    release,
                    release - resolution.at_ms,
                )
            )
    result = tuple(overlaps)
    return RenderedForegroundDissonanceAssessment(
        "assessed" if assessed else "unassessed",
        len(result),
        max((item.overlap_ms for item in result), default=0),
        result,
    )


def analyze_material_harmony(
    score: ScoreSpec,
    material_id: str,
    *,
    low_pitch_boundary: int,
    minimum_low_spacing_semitones: int,
) -> MaterialHarmonyAssessment:
    """宣言済み局所和声に対する声部、伴奏形、低音配置を診断する。"""
    if minimum_low_spacing_semitones < 1:
        raise RecurrenceQualityError("minimum low spacing must be positive")
    materials = {material.material_id: material for material in score.materials}
    if material_id not in materials:
        raise RecurrenceQualityError("harmony material does not exist")
    material = materials[material_id]
    if not material.harmonies or material.foreground_voice is None:
        return MaterialHarmonyAssessment(
            material_id,
            "unassessed",
            None,
            0,
            0,
            0,
            None,
            0,
            0,
            0,
            0,
            False,
        )
    accompaniment = "lower" if material.foreground_voice == "upper" else "upper"
    accompaniment_notes = tuple(note for note in material.notes if note.voice == accompaniment)
    foreground_notes = tuple(
        note for note in material.notes if note.voice == material.foreground_voice
    )

    def harmonies_during(note: ScoreNote):
        return tuple(
            harmony
            for harmony in material.harmonies
            if harmony.at_units < note.at_units + note.duration_units
            and note.at_units < harmony.at_units + harmony.duration_units
        )

    compliant = sum(
        bool(harmonies_during(note))
        and all(
            note.pitch % 12 in _chord_pitch_classes(harmony.root_pitch_class, harmony.quality)
            for harmony in harmonies_during(note)
        )
        for note in accompaniment_notes
    )
    chord_tone_ratio = compliant / len(accompaniment_notes) if accompaniment_notes else 0.0
    by_onset: dict[int, list[ScoreNote]] = defaultdict(list)
    for note in accompaniment_notes:
        by_onset[note.at_units].append(note)
    solo_attacks = sum(len(notes) == 1 for notes in by_onset.values())
    chord_attacks = sum(len(notes) >= 2 for notes in by_onset.values())

    arpeggio_candidates = 0
    for harmony in material.harmonies:
        start = harmony.at_units
        end = start + harmony.duration_units
        notes = tuple(note for note in accompaniment_notes if start <= note.at_units < end)
        onsets = {note.at_units for note in notes}
        pitch_classes = {
            note.pitch % 12
            for note in notes
            if note.pitch % 12 in _chord_pitch_classes(harmony.root_pitch_class, harmony.quality)
        }
        arpeggio_candidates += len(onsets) >= 3 and len(pitch_classes) >= 2

    low_spacing_violations = 0
    for onset in by_onset:
        sounding = sorted(
            {
                note.pitch
                for note in accompaniment_notes
                if note.at_units <= onset < note.at_units + note.duration_units
            }
        )
        if (
            len(sounding) >= 2
            and sounding[0] < low_pitch_boundary
            and sounding[1] - sounding[0] < minimum_low_spacing_semitones
        ):
            low_spacing_violations += 1

    foreground_non_chord = 0
    resolved_non_chord = 0
    unresolved = 0
    foreground_by_onset: dict[int, list[ScoreNote]] = defaultdict(list)
    for note in foreground_notes:
        foreground_by_onset[note.at_units].append(note)
    foreground_onsets = sorted(foreground_by_onset)
    for note in foreground_notes:
        active = harmonies_during(note)
        if active and all(
            note.pitch % 12 in _chord_pitch_classes(harmony.root_pitch_class, harmony.quality)
            for harmony in active
        ):
            continue
        foreground_non_chord += 1
        next_onset = next(
            (onset for onset in foreground_onsets if onset > note.at_units),
            None,
        )
        resolved = False
        if (
            active
            and note.at_units != active[0].at_units
            and note.duration_units * 2 <= score.divisions
            and next_onset is not None
        ):
            for target in foreground_by_onset[next_onset]:
                target_harmonies = harmonies_during(target)
                if (
                    abs(target.pitch - note.pitch) in {1, 2}
                    and target_harmonies
                    and all(
                        target.pitch % 12
                        in _chord_pitch_classes(harmony.root_pitch_class, harmony.quality)
                        for harmony in target_harmonies
                    )
                ):
                    resolved = True
                    break
        resolved_non_chord += resolved
        unresolved += not resolved

    passes = (
        bool(accompaniment_notes)
        and chord_tone_ratio == 1.0
        and low_spacing_violations == 0
        and unresolved == 0
    )
    return MaterialHarmonyAssessment(
        material.material_id,
        "assessed",
        accompaniment,
        solo_attacks,
        chord_attacks,
        arpeggio_candidates,
        chord_tone_ratio,
        low_spacing_violations,
        foreground_non_chord,
        resolved_non_chord,
        unresolved,
        passes,
    )


def _foreground_notes(material: ScoreMaterial) -> tuple[ScoreNote, ...]:
    if material.foreground_voice is None:
        raise RecurrenceQualityError("material has no declared foreground voice")
    return tuple(
        sorted(
            (note for note in material.notes if note.voice == material.foreground_voice),
            key=lambda item: (
                item.at_units,
                item.duration_units,
                item.pitch,
                item.articulations,
                item.event_id,
            ),
        )
    )


def _foreground_motif_head(material: ScoreMaterial) -> tuple[int, ...]:
    notes = _foreground_notes(material)
    by_onset: dict[int, list[int]] = defaultdict(list)
    for note in notes:
        by_onset[note.at_units].append(note.pitch)
    select = max if material.foreground_voice == "upper" else min
    melody = tuple(select(by_onset[onset]) for onset in sorted(by_onset))
    if len(melody) < 4:
        return ()
    return tuple(pitch - melody[0] for pitch in melody[:4])


def analyze_foreground_variation(
    score: ScoreSpec,
    source_material_id: str,
    target_material_id: str,
) -> ForegroundVariationAssessment:
    """宣言済み前景声部の表層差と初頭動機保持を分けて返す。"""
    materials = {material.material_id: material for material in score.materials}
    if source_material_id not in materials or target_material_id not in materials:
        raise RecurrenceQualityError("variation material does not exist")
    source = materials[source_material_id]
    target = materials[target_material_id]
    if (
        source.foreground_voice is None
        or target.foreground_voice is None
        or source.foreground_voice != target.foreground_voice
    ):
        raise RecurrenceQualityError("variation materials need the same foreground voice")
    source_notes = _foreground_notes(source)
    target_notes = _foreground_notes(target)
    pair_count = max(len(source_notes), len(target_notes))

    def item(notes: tuple[ScoreNote, ...], index: int) -> ScoreNote | None:
        return notes[index] if index < len(notes) else None

    pitch_differences = sum(
        item(source_notes, index) is None
        or item(target_notes, index) is None
        or item(source_notes, index).pitch != item(target_notes, index).pitch
        for index in range(pair_count)
    )
    rhythm_differences = sum(
        item(source_notes, index) is None
        or item(target_notes, index) is None
        or (
            item(source_notes, index).at_units,
            item(source_notes, index).duration_units,
        )
        != (
            item(target_notes, index).at_units,
            item(target_notes, index).duration_units,
        )
        for index in range(pair_count)
    )
    articulation_differences = sum(
        item(source_notes, index) is None
        or item(target_notes, index) is None
        or item(source_notes, index).articulations != item(target_notes, index).articulations
        for index in range(pair_count)
    )
    exact = (
        len(source_notes) == len(target_notes)
        and pitch_differences == 0
        and rhythm_differences == 0
        and articulation_differences == 0
    )
    source_head = _foreground_motif_head(source)
    target_head = _foreground_motif_head(target)
    head_preserved = bool(source_head) and source_head == target_head
    return ForegroundVariationAssessment(
        source.material_id,
        target.material_id,
        source.foreground_voice,
        len(source_notes),
        len(target_notes),
        pitch_differences,
        rhythm_differences,
        articulation_differences,
        head_preserved,
        exact,
        not exact and head_preserved,
    )


def _node_voice_notes(
    node: PlanNode,
    children: dict[str, list[PlanNode]],
    materials: dict[str, ScoreMaterial],
    voice: str,
) -> tuple[tuple[int, ScoreNote], ...]:
    cursor = 0
    result: list[tuple[int, ScoreNote]] = []
    for leaf in _subtree_leaves(node, children):
        assert leaf.score_material_id is not None
        material = materials[leaf.score_material_id]
        result.extend(
            (cursor + note.at_units, note) for note in material.notes if note.voice == voice
        )
        cursor += material.length_units
    return tuple(sorted(result, key=lambda item: (item[0], item[1].pitch)))


def analyze_transition_connection(
    plan: PiecePlan,
    score: ScoreSpec,
    *,
    source_node_id: str,
    transition_node_id: str,
    target_node_id: str,
    voice: str,
    minimum_transition_attacks: int,
    maximum_boundary_leap: int,
    maximum_internal_leap: int,
) -> TransitionConnectionAssessment:
    """明示された接続句の境界音と内部輪郭を楽譜時間で診断する。"""
    if voice not in {"upper", "lower"}:
        raise RecurrenceQualityError("transition voice is invalid")
    if minimum_transition_attacks < 2 or min(maximum_boundary_leap, maximum_internal_leap) < 0:
        raise RecurrenceQualityError("transition thresholds are invalid")
    nodes = {node.node_id: node for node in plan.nodes}
    if any(
        node_id not in nodes for node_id in (source_node_id, transition_node_id, target_node_id)
    ):
        raise RecurrenceQualityError("transition node does not exist")
    children = _children(plan)
    materials = {material.material_id: material for material in score.materials}
    source_notes = _node_voice_notes(nodes[source_node_id], children, materials, voice)
    transition_notes = _node_voice_notes(nodes[transition_node_id], children, materials, voice)
    target_notes = _node_voice_notes(nodes[target_node_id], children, materials, voice)
    if not source_notes or not transition_notes or not target_notes:
        raise RecurrenceQualityError("transition comparison needs notes in the selected voice")

    def representative(
        notes: tuple[tuple[int, ScoreNote], ...],
    ) -> tuple[tuple[int, int], ...]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for onset, note in notes:
            grouped[onset].append(note.pitch)
        select = max if voice == "upper" else min
        return tuple((onset, select(grouped[onset])) for onset in sorted(grouped))

    source_line = representative(source_notes)
    transition_line = representative(transition_notes)
    target_line = representative(target_notes)
    source_leap = abs(source_line[-1][1] - transition_line[0][1])
    internal_leap = max(
        (abs(right[1] - left[1]) for left, right in pairwise(transition_line)),
        default=0,
    )
    target_leap = abs(transition_line[-1][1] - target_line[0][1])
    restrike = transition_line[-1][1] == target_line[0][1]
    passes = (
        len(transition_line) >= minimum_transition_attacks
        and source_leap <= maximum_boundary_leap
        and internal_leap <= maximum_internal_leap
        and target_leap <= maximum_boundary_leap
        and not restrike
    )
    return TransitionConnectionAssessment(
        source_node_id,
        transition_node_id,
        target_node_id,
        voice,
        len(transition_line),
        source_leap,
        internal_leap,
        target_leap,
        restrike,
        passes,
    )


def analyze_rendered_harmony(
    plan: PiecePlan,
    score: ScoreSpec,
    rendered: RenderedPerformance,
) -> RenderedHarmonyAssessment:
    """ペダル保持で旧伴奏音が次の局所和声へ残る回数を返す。"""
    del plan, score
    if not rendered.harmonies:
        return RenderedHarmonyAssessment("unassessed", 0, False)
    pedals = sorted(rendered.pedals, key=lambda item: (item.at_ms, item.value))

    def pedal_down_at(at_ms: int) -> bool:
        state = 0
        for pedal in pedals:
            if pedal.at_ms > at_ms:
                break
            state = pedal.value
        return state >= 64

    def actually_sounding(note, at_ms: int) -> bool:
        note_off = note.at_ms + note.duration_ms
        if note.at_ms < at_ms < note_off:
            return True
        if not note.at_ms < note_off <= at_ms or not pedal_down_at(note_off):
            return False
        return not any(pedal.value < 64 and note_off <= pedal.at_ms <= at_ms for pedal in pedals)

    violations = 0
    for harmony in rendered.harmonies:
        if not any(other.end_ms == harmony.start_ms for other in rendered.harmonies):
            continue
        chord = _chord_pitch_classes(harmony.root_pitch_class, harmony.quality)
        violations += sum(
            note.voice == harmony.accompaniment_voice
            and note.at_ms < harmony.start_ms
            and actually_sounding(note, harmony.start_ms)
            and note.pitch % 12 not in chord
            for note in rendered.notes
        )
    return RenderedHarmonyAssessment("assessed", violations, violations == 0)


def analyze_rendered_boundary(
    plan: PiecePlan,
    rendered: RenderedPerformance,
    *,
    transition_node_id: str,
    target_node_id: str,
    voice: str,
    maximum_pedal_gap_ms: int,
) -> RenderedBoundaryAssessment:
    """接続句末尾から次区分先頭までの打鍵とペダル状態を診断する。"""
    if voice not in {"upper", "lower"} or maximum_pedal_gap_ms < 0:
        raise RecurrenceQualityError("rendered boundary parameters are invalid")
    nodes = {node.node_id: node for node in plan.nodes}
    if transition_node_id not in nodes or target_node_id not in nodes:
        raise RecurrenceQualityError("rendered boundary node does not exist")
    children = _children(plan)
    transition_leaves = {
        leaf.node_id for leaf in _subtree_leaves(nodes[transition_node_id], children)
    }
    target_leaves = {leaf.node_id for leaf in _subtree_leaves(nodes[target_node_id], children)}
    transition_notes = tuple(
        note
        for note in rendered.notes
        if note.occurrence_node_id in transition_leaves and note.voice == voice
    )
    target_notes = tuple(
        note
        for note in rendered.notes
        if note.occurrence_node_id in target_leaves and note.voice == voice
    )
    if not transition_notes or not target_notes:
        raise RecurrenceQualityError("rendered boundary needs notes in the selected voice")
    transition_attack = max(note.at_ms for note in transition_notes)
    transition_last_notes = tuple(
        note for note in transition_notes if note.at_ms == transition_attack
    )
    target_attack = min(note.at_ms for note in target_notes)
    target_first_notes = tuple(note for note in target_notes if note.at_ms == target_attack)
    note_off = max(note.at_ms + note.duration_ms for note in transition_last_notes)
    note_gap = max(0, target_attack - note_off)
    restrike = bool(
        {note.pitch for note in transition_last_notes} & {note.pitch for note in target_first_notes}
    )

    pedals = tuple(sorted(rendered.pedals, key=lambda item: (item.at_ms, item.value)))
    before = tuple(pedal for pedal in pedals if pedal.at_ms < target_attack)
    state_before = before[-1].value if before else 0
    if state_before >= 64 and not any(
        pedal.at_ms == target_attack and pedal.value < 64 for pedal in pedals
    ):
        pedal_gap = 0
    else:
        releases = tuple(
            pedal.at_ms for pedal in pedals if pedal.at_ms <= target_attack and pedal.value < 64
        )
        release = max(releases, default=target_attack)
        downs = tuple(
            pedal.at_ms for pedal in pedals if pedal.at_ms >= release and pedal.value >= 64
        )
        down = min(downs, default=target_attack + maximum_pedal_gap_ms + 1)
        pedal_gap = max(0, down - release)
    return RenderedBoundaryAssessment(
        transition_node_id,
        target_node_id,
        voice,
        transition_attack,
        target_attack,
        note_gap,
        restrike,
        pedal_gap,
        not restrike and pedal_gap <= maximum_pedal_gap_ms,
    )


def analyze_material_voice_texture(material: ScoreMaterial) -> VoiceTextureAssessment:
    """共有する最初と最後の発音、およびその間の声部独立を返す。"""
    upper = {note.at_units for note in material.notes if note.voice == "upper"}
    lower = {note.at_units for note in material.notes if note.voice == "lower"}
    shared = sorted(upper & lower)
    opening = shared[0] if shared else None
    closing = shared[-1] if shared else None
    independent = (
        opening is not None
        and closing is not None
        and opening < closing
        and any(opening < onset < closing for onset in upper ^ lower)
    )
    return VoiceTextureAssessment(
        material_id=material.material_id,
        opening_anchor=opening,
        closing_anchor=closing,
        internal_independence=independent,
        passes=opening is not None and closing is not None and independent,
    )


def analyze_material_vertical_alignment(
    material: ScoreMaterial,
    *,
    minimum_ratio: float,
) -> MaterialAlignmentAssessment:
    """楽譜素材内の上下声の共有打鍵率と、共有打鍵間の独立性を返す。"""
    if not 0 <= minimum_ratio <= 1:
        raise RecurrenceQualityError("material alignment ratio must be between zero and one")
    upper = {note.at_units for note in material.notes if note.voice == "upper"}
    lower = {note.at_units for note in material.notes if note.voice == "lower"}
    shared = sorted(upper & lower)
    denominator = len(upper) + len(lower)
    ratio = 2 * len(shared) / denominator if denominator else 0.0
    independent = len(shared) >= 2 and any(
        shared[0] < onset < shared[-1] for onset in upper ^ lower
    )
    return MaterialAlignmentAssessment(
        material_id=material.material_id,
        shared_attack_ratio=ratio,
        internal_independence=independent,
        minimum_ratio=minimum_ratio,
        passes=ratio >= minimum_ratio and independent,
    )


def analyze_piano_texture_variety(
    material: ScoreMaterial,
    *,
    minimum_maximum_polyphony: int = 3,
) -> PianoTextureAssessment:
    """発音位置ごとの打鍵数と、楽譜上の最大同時発音数を返す。"""
    if minimum_maximum_polyphony < 1:
        raise RecurrenceQualityError("minimum maximum polyphony must be positive")
    attack_sizes: dict[int, int] = defaultdict(int)
    for note in material.notes:
        attack_sizes[note.at_units] += 1
    size_counts: dict[int, int] = defaultdict(int)
    for size in attack_sizes.values():
        size_counts[size] += 1
    maximum_polyphony = max(
        (
            sum(
                note.at_units <= onset < note.at_units + note.duration_units
                for note in material.notes
            )
            for onset in attack_sizes
        ),
        default=0,
    )
    has_single_attack = 1 in size_counts
    has_multiple_attack = any(size >= 2 for size in size_counts)
    return PianoTextureAssessment(
        material_id=material.material_id,
        attack_size_counts=tuple(sorted(size_counts.items())),
        maximum_polyphony=maximum_polyphony,
        passes=(
            has_single_attack
            and has_multiple_attack
            and maximum_polyphony >= minimum_maximum_polyphony
        ),
    )


def _children(plan: PiecePlan) -> dict[str, list[PlanNode]]:
    result: dict[str, list[PlanNode]] = defaultdict(list)
    for node in plan.nodes:
        if node.parent_id is not None:
            result[node.parent_id].append(node)
    for items in result.values():
        items.sort(key=lambda item: item.order)
    return result


def _subtree_leaves(
    node: PlanNode,
    children: dict[str, list[PlanNode]],
) -> tuple[PlanNode, ...]:
    if not children[node.node_id]:
        return (node,)
    return tuple(
        leaf for child in children[node.node_id] for leaf in _subtree_leaves(child, children)
    )


def _score_features(
    node: PlanNode,
    children: dict[str, list[PlanNode]],
    materials: dict[str, ScoreMaterial],
    feature_voice: str | None = None,
) -> dict[str, object]:
    rhythm_grid: set[Fraction] = set()
    pitch_classes: set[int] = set()
    pitches: list[int] = []
    upper_attacks = 0
    lower_attacks = 0
    articulation_count = 0
    feature_note_count = 0
    chord_attacks = 0
    attack_count = 0
    for leaf in _subtree_leaves(node, children):
        assert leaf.score_material_id is not None
        material = materials[leaf.score_material_id]
        by_onset: dict[int, list[ScoreNote]] = defaultdict(list)
        upper_onsets: set[int] = set()
        lower_onsets: set[int] = set()
        for note in material.notes:
            by_onset[note.at_units].append(note)
            pitch_classes.add(note.pitch % 12)
            pitches.append(note.pitch)
            if feature_voice is None or note.voice == feature_voice:
                rhythm_grid.add(Fraction(note.at_units, material.length_units))
                articulation_count += bool({"accent", "staccato"} & set(note.articulations))
                feature_note_count += 1
            if note.voice == "upper":
                upper_onsets.add(note.at_units)
            else:
                lower_onsets.add(note.at_units)
        upper_attacks += len(upper_onsets)
        lower_attacks += len(lower_onsets)
        attack_count += len(by_onset)
        chord_attacks += sum(len(notes) >= 2 for notes in by_onset.values())
    return {
        "rhythm_grid": rhythm_grid,
        "pitch_classes": pitch_classes,
        "register": statistics.median(pitches),
        "upper_share": upper_attacks / (upper_attacks + lower_attacks),
        "articulation_ratio": (
            articulation_count / feature_note_count if feature_note_count else 0.0
        ),
        "chord_ratio": chord_attacks / attack_count,
    }


def analyze_section_contrast(
    plan: PiecePlan,
    score: ScoreSpec,
    *,
    target_node_id: str,
    minimum_changed_axes: int = 3,
    minimum_shared_pitch_class_ratio: float = 0.5,
    feature_voice: str | None = None,
) -> SectionContrastAssessment:
    """明示された対比を複数の場面差軸と共通音高クラスへ分けて診断する。"""
    if feature_voice not in {None, "upper", "lower"}:
        raise RecurrenceQualityError("contrast feature voice is invalid")
    nodes = {node.node_id: node for node in plan.nodes}
    if target_node_id not in nodes or nodes[target_node_id].contrasts_with is None:
        raise RecurrenceQualityError("contrast target is not declared")
    target = nodes[target_node_id]
    source = nodes[target.contrasts_with]
    children = _children(plan)
    materials = {material.material_id: material for material in score.materials}
    source_features = _score_features(source, children, materials, feature_voice)
    target_features = _score_features(target, children, materials, feature_voice)
    source_grid = source_features["rhythm_grid"]
    target_grid = target_features["rhythm_grid"]
    assert isinstance(source_grid, set) and isinstance(target_grid, set)
    if not source_grid or not target_grid:
        raise RecurrenceQualityError("contrast feature voice has no note attacks")
    rhythm_distance = 1.0 - len(source_grid & target_grid) / len(source_grid | target_grid)
    source_share = float(source_features["upper_share"])
    target_share = float(target_features["upper_share"])
    changed: list[str] = []
    if rhythm_distance >= 0.35:
        changed.append("rhythm_grid")
    if abs(source_share - target_share) >= 0.20 and (source_share >= 0.5) != (target_share >= 0.5):
        changed.append("hand_role")
    if abs(float(source_features["register"]) - float(target_features["register"])) >= 5:
        changed.append("register")
    articulation_delta = abs(
        float(source_features["articulation_ratio"]) - float(target_features["articulation_ratio"])
    )
    if articulation_delta >= 0.15:
        changed.append("articulation")
    if abs(float(source_features["chord_ratio"]) - float(target_features["chord_ratio"])) >= 0.15:
        changed.append("chord_shape")
    source_pitch_classes = source_features["pitch_classes"]
    target_pitch_classes = target_features["pitch_classes"]
    assert isinstance(source_pitch_classes, set) and isinstance(target_pitch_classes, set)
    shared_ratio = len(source_pitch_classes & target_pitch_classes) / len(
        source_pitch_classes | target_pitch_classes
    )
    return SectionContrastAssessment(
        target_node_id=target.node_id,
        source_node_id=source.node_id,
        changed_axes=tuple(changed),
        rhythm_grid_distance=rhythm_distance,
        articulation_ratio_delta=articulation_delta,
        shared_pitch_class_ratio=shared_ratio,
        passes=(
            len(changed) >= minimum_changed_axes
            and shared_ratio >= minimum_shared_pitch_class_ratio
        ),
    )


def _performed_alignment_ratio(
    node: PlanNode,
    children: dict[str, list[PlanNode]],
    rendered: RenderedPerformance,
    tolerance_ms: int,
) -> float:
    leaf_ids = {leaf.node_id for leaf in _subtree_leaves(node, children)}
    upper = {
        note.at_ms
        for note in rendered.notes
        if note.occurrence_node_id in leaf_ids and note.voice == "upper"
    }
    lower = {
        note.at_ms
        for note in rendered.notes
        if note.occurrence_node_id in leaf_ids and note.voice == "lower"
    }
    if not upper or not lower:
        return 0.0
    matched_upper = sum(
        any(abs(at_ms - other) <= tolerance_ms for other in lower) for at_ms in upper
    )
    matched_lower = sum(
        any(abs(at_ms - other) <= tolerance_ms for other in upper) for at_ms in lower
    )
    return (matched_upper + matched_lower) / (len(upper) + len(lower))


def analyze_vertical_alignment(
    plan: PiecePlan,
    rendered: RenderedPerformance,
    *,
    source_node_id: str,
    target_node_id: str,
    minimum_target_ratio: float,
    minimum_increase: float,
    tolerance_ms: int = 10,
) -> VerticalAlignmentAssessment:
    """二つの部分木で上下声の実演奏上の同時打鍵率を比較する。"""
    if minimum_target_ratio < 0 or minimum_increase < 0 or tolerance_ms < 0:
        raise RecurrenceQualityError("alignment thresholds must not be negative")
    nodes = {node.node_id: node for node in plan.nodes}
    if source_node_id not in nodes or target_node_id not in nodes:
        raise RecurrenceQualityError("alignment node does not exist")
    children = _children(plan)
    source_ratio = _performed_alignment_ratio(
        nodes[source_node_id], children, rendered, tolerance_ms
    )
    target_ratio = _performed_alignment_ratio(
        nodes[target_node_id], children, rendered, tolerance_ms
    )
    return VerticalAlignmentAssessment(
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        source_ratio=source_ratio,
        target_ratio=target_ratio,
        minimum_target_ratio=minimum_target_ratio,
        minimum_increase=minimum_increase,
        passes=(
            target_ratio >= minimum_target_ratio and target_ratio - source_ratio >= minimum_increase
        ),
    )


def analyze_section_pacing(
    plan: PiecePlan,
    score: ScoreSpec,
    rendered: RenderedPerformance,
    *,
    source_node_id: str,
    target_node_id: str,
    maximum_target_ratio: float,
) -> SectionPacingAssessment:
    """二つの構成ノードを、発音数と独立した1楽譜単位当たりの時間で比べる。"""
    if maximum_target_ratio <= 0:
        raise RecurrenceQualityError("maximum target ratio must be positive")
    node_by_id = {node.node_id: node for node in plan.nodes}
    if source_node_id not in node_by_id or target_node_id not in node_by_id:
        raise RecurrenceQualityError("pacing node does not exist")
    interval_by_id = {
        node_id: (start_ms, end_ms) for node_id, start_ms, end_ms in rendered.node_intervals
    }
    if source_node_id not in interval_by_id or target_node_id not in interval_by_id:
        raise RecurrenceQualityError("rendered node interval does not exist")
    children = _children(plan)
    materials = {material.material_id: material for material in score.materials}

    def measurements(node_id: str) -> tuple[float, int]:
        leaves = _subtree_leaves(node_by_id[node_id], children)
        units = sum(materials[leaf.score_material_id].length_units for leaf in leaves)
        start_ms, end_ms = interval_by_id[node_id]
        leaf_ids = {leaf.node_id for leaf in leaves}
        note_count = sum(note.occurrence_node_id in leaf_ids for note in rendered.notes)
        return (end_ms - start_ms) / units, note_count

    source_ms_per_unit, source_note_count = measurements(source_node_id)
    target_ms_per_unit, target_note_count = measurements(target_node_id)
    return SectionPacingAssessment(
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        source_ms_per_unit=source_ms_per_unit,
        target_ms_per_unit=target_ms_per_unit,
        source_note_count=source_note_count,
        target_note_count=target_note_count,
        maximum_target_ratio=maximum_target_ratio,
        passes=target_ms_per_unit / source_ms_per_unit <= maximum_target_ratio,
    )


def analyze_local_pulse(
    plan: PiecePlan,
    score: ScoreSpec,
    rendered: RenderedPerformance,
    *,
    node_id: str,
    voice: str,
    minimum_variation_ratio: float,
    maximum_adjacent_ratio: float,
) -> LocalPulseAssessment:
    """楽譜間隔で正規化した一声部の局所脈動と同時打鍵の広がりを返す。"""
    if voice not in {"upper", "lower"}:
        raise RecurrenceQualityError("pulse voice is invalid")
    if minimum_variation_ratio < 1 or maximum_adjacent_ratio < 1:
        raise RecurrenceQualityError("pulse ratios must be at least one")
    nodes = {node.node_id: node for node in plan.nodes}
    if node_id not in nodes:
        raise RecurrenceQualityError("pulse node does not exist")
    children = _children(plan)
    materials = {material.material_id: material for material in score.materials}
    leaf_starts: dict[str, int] = {}
    cursor = 0

    def schedule(node: PlanNode) -> None:
        nonlocal cursor
        descendants = children[node.node_id]
        if descendants:
            for child in descendants:
                schedule(child)
            return
        assert node.score_material_id is not None
        leaf_starts[node.node_id] = cursor
        cursor += materials[node.score_material_id].length_units

    schedule(nodes[plan.root_node_id])
    performed = {(note.occurrence_node_id, note.event_id): note.at_ms for note in rendered.notes}
    attacks: dict[int, list[int]] = defaultdict(list)
    for leaf in _subtree_leaves(nodes[node_id], children):
        assert leaf.score_material_id is not None
        material = materials[leaf.score_material_id]
        for note in material.notes:
            if note.voice != voice:
                continue
            event_id = f"{leaf.node_id}:{note.event_id}"
            key = (leaf.node_id, event_id)
            if key not in performed:
                raise RecurrenceQualityError("rendered pulse event does not exist")
            attacks[leaf_starts[leaf.node_id] + note.at_units].append(performed[key])
    ordered = sorted(attacks)
    if len(ordered) < 3:
        raise RecurrenceQualityError("pulse node needs at least three attack positions")
    maximum_spread = max(max(attacks[unit]) - min(attacks[unit]) for unit in ordered)
    attack_times = {unit: round(statistics.median(attacks[unit])) for unit in ordered}
    local_rates = tuple(
        (attack_times[right] - attack_times[left]) / (right - left)
        for left, right in pairwise(ordered)
    )
    if any(rate <= 0 for rate in local_rates):
        raise RecurrenceQualityError("pulse time must increase between attack positions")
    adjacent_ratios = tuple(
        max(left / right, right / left) for left, right in pairwise(local_rates)
    )
    minimum_rate = min(local_rates)
    maximum_rate = max(local_rates)
    variation = maximum_rate / minimum_rate
    maximum_local_change = max(adjacent_ratios, default=1.0)
    return LocalPulseAssessment(
        node_id=node_id,
        voice=voice,
        interval_count=len(local_rates),
        minimum_ms_per_unit=minimum_rate,
        median_ms_per_unit=statistics.median(local_rates),
        maximum_ms_per_unit=maximum_rate,
        variation_ratio=variation,
        maximum_adjacent_ratio=maximum_local_change,
        simultaneous_attack_max_spread_ms=maximum_spread,
        minimum_variation_ratio=minimum_variation_ratio,
        maximum_allowed_adjacent_ratio=maximum_adjacent_ratio,
        passes=(
            variation >= minimum_variation_ratio
            and maximum_local_change <= maximum_adjacent_ratio
            and maximum_spread == 0
        ),
    )


def analyze_boundary_breath(
    plan: PiecePlan,
    score: ScoreSpec,
    node_id: str,
    *,
    minimum_gap_units: int,
) -> BoundaryBreathAssessment:
    """部分木末尾の新規発音間隔を、呼出し側の校正閾値で判定する。"""
    if minimum_gap_units < 0:
        raise RecurrenceQualityError("minimum gap must not be negative")
    node_by_id = {node.node_id: node for node in plan.nodes}
    if node_id not in node_by_id:
        raise RecurrenceQualityError("boundary node does not exist")
    children = _children(plan)
    materials = {material.material_id: material for material in score.materials}
    cursor = 0
    onsets: list[int] = []

    def visit(node: PlanNode) -> None:
        nonlocal cursor
        descendants = children[node.node_id]
        if descendants:
            for child in descendants:
                visit(child)
            return
        assert node.score_material_id is not None
        material = materials[node.score_material_id]
        onsets.extend(cursor + note.at_units for note in material.notes)
        cursor += material.length_units

    visit(node_by_id[node_id])
    if not onsets:
        raise RecurrenceQualityError("boundary subtree has no note attacks")
    gap = cursor - max(onsets)
    return BoundaryBreathAssessment(
        node_id=node_id,
        new_attack_gap_units=gap,
        minimum_gap_units=minimum_gap_units,
        passes=gap >= minimum_gap_units,
    )
