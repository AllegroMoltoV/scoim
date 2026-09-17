"""曲全体の調性計画から、あかるさの暗い側と明るい側を作る。"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace

from llm_musical_composer.control_axis_anchors import fixed_tonic_brightness_metrics
from llm_musical_composer.performance_pipeline import (
    PerformanceSpec,
    PiecePlan,
    RenderedPerformance,
    ScoreMaterial,
    ScoreSpec,
    harmony_pitch_classes,
)
from llm_musical_composer.recurrence_quality import analyze_foreground_dissonance

SCHEME_ID = "brightness-v2-whole-song-tonal-endpoints"
_INTERVALS = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "diminished": (0, 3, 6),
}


def _chord(root: int, quality: str) -> tuple[int, str]:
    return root, quality


_AM = _chord(9, "minor")
_A = _chord(9, "major")
_C = _chord(0, "major")
_DM = _chord(2, "minor")
_D = _chord(2, "major")
_E = _chord(4, "major")
_F = _chord(5, "major")
_G = _chord(7, "major")

_TONAL_PLANS: dict[str, dict[str, tuple[tuple[int, str], ...]]] = {
    "dark-primary": {
        "a-theme": (_AM, _DM, _F, _E),
        "a-theme-prime": (_AM, _F, _DM, _E),
        "a-contrast": (_F, _C, _DM, _E),
        "a-contrast-prime": (_F, _DM, _C, _E),
        "a-return": (_AM, _DM, _F, _E),
        "a-return-prime": (_AM, _F, _DM, _E),
        "b-theme": (_C, _F, _AM, _E),
        "b-theme-prime": (_C, _DM, _AM, _E),
        "b-contrast": (_F, _C, _G, _E),
        "b-contrast-prime": (_F, _DM, _G, _E),
        "b-return": (_C, _F, _AM, _E),
        "b-return-prime": (_C, _DM, _AM, _E),
        "transition-ab": (_C, _AM),
        "transition-ba2": (_DM, _DM),
    },
    "bright-primary": {
        "a-theme": (_A, _D, _A, _E),
        "a-theme-prime": (_A, _A, _D, _E),
        "a-contrast": (_D, _A, _D, _E),
        "a-contrast-prime": (_D, _D, _A, _E),
        "a-return": (_A, _D, _A, _E),
        "a-return-prime": (_A, _A, _D, _E),
        "b-theme": (_A, _D, _E, _A),
        "b-theme-prime": (_A, _E, _D, _A),
        "b-contrast": (_D, _A, _E, _A),
        "b-contrast-prime": (_D, _E, _A, _A),
        "b-return": (_A, _D, _E, _A),
        "b-return-prime": (_A, _E, _D, _A),
        "transition-ab": (_A, _D),
        "transition-ba2": (_D, _D),
    },
}


@dataclass(frozen=True)
class TransitionPath:
    material_id: str
    pitches: tuple[int, ...]
    cost: int


@dataclass(frozen=True)
class BrightnessVariant:
    level: str
    requested: int
    tonal_plan_id: str
    plan: PiecePlan
    score: ScoreSpec
    performance: PerformanceSpec
    repaired_event_ids: tuple[str, ...] = ()
    transition_paths: tuple[TransitionPath, ...] = ()


def audit_score_layout(score: ScoreSpec) -> dict[str, list[str]]:
    """和声区間数ごとに素材を列挙し、変換契約の前提を可視化する。"""
    result = {
        "four_harmony_material_ids": [],
        "two_harmony_material_ids": [],
        "harmony_free_material_ids": [],
    }
    for material in score.materials:
        if len(material.harmonies) == 4:
            result["four_harmony_material_ids"].append(material.material_id)
        elif len(material.harmonies) == 2:
            result["two_harmony_material_ids"].append(material.material_id)
        elif not material.harmonies:
            result["harmony_free_material_ids"].append(material.material_id)
        else:
            raise ValueError(
                "brightness v2 does not support "
                f"{len(material.harmonies)} harmony spans: {material.material_id}"
            )
    return result


def _material_role(material_id: str) -> str:
    if material_id.startswith(("a1-", "a2-")):
        return "a-" + material_id.split("-", 1)[1]
    return material_id


def _nearest_pitch(pitch: int, pitch_class: int) -> int:
    candidates = tuple(candidate for candidate in range(21, 109) if candidate % 12 == pitch_class)
    return min(candidates, key=lambda candidate: (abs(candidate - pitch), candidate))


def _chord_roles(root: int, quality: str) -> tuple[int, ...]:
    return tuple((root + interval) % 12 for interval in _INTERVALS[quality])


def _harmony_index(material: ScoreMaterial, at_units: int) -> int:
    for index, harmony in enumerate(material.harmonies):
        if harmony.at_units <= at_units < harmony.at_units + harmony.duration_units:
            return index
    raise ValueError(f"note onset is outside harmony coverage: {material.material_id}:{at_units}")


def _transform_harmonic_material(
    material: ScoreMaterial,
    targets: tuple[tuple[int, str], ...],
    *,
    mode: str,
) -> ScoreMaterial:
    target_harmonies = tuple(
        replace(source, root_pitch_class=root, quality=quality)
        for source, (root, quality) in zip(material.harmonies, targets, strict=True)
    )
    parallel_map = {0: 1, 5: 6, 7: 8} if mode == "major" else {1: 0, 6: 5, 8: 7}
    notes = []
    for note in material.notes:
        index = _harmony_index(material, note.at_units)
        source = material.harmonies[index]
        target = target_harmonies[index]
        source_roles = _chord_roles(source.root_pitch_class, source.quality)
        target_roles = _chord_roles(target.root_pitch_class, target.quality)
        if note.pitch % 12 in source_roles:
            role_index = source_roles.index(note.pitch % 12)
            target_pitch_class = target_roles[role_index]
        else:
            target_pitch_class = parallel_map.get(note.pitch % 12, note.pitch % 12)
        notes.append(replace(note, pitch=_nearest_pitch(note.pitch, target_pitch_class)))
    return replace(material, harmonies=target_harmonies, notes=tuple(notes))


def _transform_harmony_free_material(material: ScoreMaterial, *, mode: str) -> ScoreMaterial:
    parallel_map = {0: 1, 5: 6, 7: 8} if mode == "major" else {1: 0, 6: 5, 8: 7}
    return replace(
        material,
        notes=tuple(
            replace(
                note,
                pitch=_nearest_pitch(
                    note.pitch,
                    parallel_map.get(note.pitch % 12, note.pitch % 12),
                ),
            )
            for note in material.notes
        ),
    )


def _preserve_foreground_roles(
    base: ScoreSpec,
    transformed: ScoreSpec,
) -> tuple[ScoreSpec, tuple[str, ...]]:
    """変換前の経過音と隣接音の前後関係を、成立する場合だけ保つ。"""
    base_materials = {material.material_id: material for material in base.materials}
    repaired_event_ids: list[str] = []
    materials: list[ScoreMaterial] = []
    for material in transformed.materials:
        if not material.harmonies or material.foreground_voice is None:
            materials.append(material)
            continue
        functional_tones = {
            tone.event_id: tone
            for tone in analyze_foreground_dissonance(base, material.material_id).tones
            if tone.role in {"passing", "neighbor"}
        }
        if not functional_tones:
            materials.append(material)
            continue
        base_material = base_materials[material.material_id]
        by_onset: dict[int, list] = defaultdict(list)
        for note in base_material.notes:
            if note.voice == base_material.foreground_voice:
                by_onset[note.at_units].append(note)
        select = max if base_material.foreground_voice == "upper" else min
        representatives = tuple(
            select(by_onset[onset], key=lambda note: note.pitch) for onset in sorted(by_onset)
        )
        representative_index = {note.event_id: index for index, note in enumerate(representatives)}
        transformed_notes = {note.event_id: note for note in material.notes}
        replacements: dict[str, int] = {}
        for event_id, tone in functional_tones.items():
            index = representative_index[event_id]
            if index == 0 or index + 1 >= len(representatives):
                continue
            if tone.approach_semitones is None or tone.departure_semitones is None:
                continue
            previous = transformed_notes[representatives[index - 1].event_id]
            following = transformed_notes[representatives[index + 1].event_id]
            candidate = previous.pitch + tone.approach_semitones
            if candidate + tone.departure_semitones != following.pitch:
                continue
            if not 0 <= candidate <= 127:
                continue
            if transformed_notes[event_id].pitch != candidate:
                replacements[event_id] = candidate
                repaired_event_ids.append(event_id)
        materials.append(
            replace(
                material,
                notes=tuple(
                    replace(note, pitch=replacements[note.event_id])
                    if note.event_id in replacements
                    else note
                    for note in material.notes
                ),
            )
        )
    return replace(transformed, materials=tuple(materials)), tuple(sorted(repaired_event_ids))


def _repair_unsupported(score: ScoreSpec) -> tuple[ScoreSpec, tuple[str, ...]]:
    repaired_ids = []
    materials = []
    for material in score.materials:
        if not material.harmonies or material.material_id.startswith("transition-"):
            materials.append(material)
            continue
        unsupported = {
            tone.event_id
            for tone in analyze_foreground_dissonance(score, material.material_id).tones
            if tone.role == "unsupported"
        }
        notes = []
        for note in material.notes:
            if note.event_id not in unsupported:
                notes.append(note)
                continue
            harmony = material.harmonies[_harmony_index(material, note.at_units)]
            pitch_classes = harmony_pitch_classes(harmony.root_pitch_class, harmony.quality)
            pitch = min(
                (_nearest_pitch(note.pitch, pitch_class) for pitch_class in pitch_classes),
                key=lambda candidate: (abs(candidate - note.pitch), candidate),
            )
            notes.append(replace(note, pitch=pitch))
            repaired_ids.append(note.event_id)
        materials.append(replace(material, notes=tuple(notes)))
    return replace(score, materials=tuple(materials)), tuple(sorted(repaired_ids))


def _upper_notes(material: ScoreMaterial):
    return tuple(
        sorted(
            (note for note in material.notes if note.voice == "upper"),
            key=lambda note: (note.at_units, note.event_id),
        )
    )


def _boundary_pitch(material: ScoreMaterial, *, last: bool) -> int:
    upper = _upper_notes(material)
    onset = (max if last else min)(note.at_units for note in upper)
    return max(note.pitch for note in upper if note.at_units == onset)


def _transition_path(
    material: ScoreMaterial,
    source: ScoreMaterial,
    target: ScoreMaterial,
) -> tuple[ScoreMaterial, TransitionPath]:
    upper = _upper_notes(material)
    source_last = _boundary_pitch(source, last=True)
    target_first = _boundary_pitch(target, last=False)
    layers = []
    for note in upper:
        harmony = material.harmonies[_harmony_index(material, note.at_units)]
        pitch_classes = harmony_pitch_classes(harmony.root_pitch_class, harmony.quality)
        layers.append(tuple(pitch for pitch in range(21, 109) if pitch % 12 in pitch_classes))
    state = {
        pitch: (abs(pitch - upper[0].pitch), (pitch,))
        for pitch in layers[0]
        if abs(pitch - source_last) <= 2 and pitch != source_last
    }
    for index in range(1, len(layers)):
        next_state = {}
        original_step = upper[index].pitch - upper[index - 1].pitch
        for pitch in layers[index]:
            choices = [
                (
                    cost
                    + abs(pitch - upper[index].pitch)
                    + abs((pitch - previous) - original_step),
                    (*path, pitch),
                )
                for previous, (cost, path) in state.items()
                if abs(pitch - previous) <= 5 and pitch != previous
            ]
            if choices:
                next_state[pitch] = min(choices, key=lambda item: (item[0], item[1]))
        state = next_state
    valid = [
        value
        for pitch, value in state.items()
        if abs(pitch - target_first) <= 2 and pitch != target_first
    ]
    if not valid:
        raise ValueError(
            f"no transition path: {material.material_id}; "
            f"source={source_last}; target={target_first}"
        )
    cost, path = min(valid, key=lambda item: (item[0], item[1]))
    replacements = {note.event_id: pitch for note, pitch in zip(upper, path, strict=True)}
    transformed = replace(
        material,
        notes=tuple(
            replace(note, pitch=replacements[note.event_id])
            if note.event_id in replacements
            else note
            for note in material.notes
        ),
    )
    return transformed, TransitionPath(material.material_id, path, cost)


def _build_variant(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
    *,
    level: str,
    scheme_id: str,
) -> BrightnessVariant:
    mode = "minor" if level == "low" else "major"
    table = _TONAL_PLANS[scheme_id]
    materials = []
    for material in score.materials:
        if material.harmonies:
            materials.append(
                _transform_harmonic_material(
                    material,
                    table[_material_role(material.material_id)],
                    mode=mode,
                )
            )
        else:
            materials.append(_transform_harmony_free_material(material, mode=mode))
    transformed = replace(
        score,
        score_id=f"{score.score_id}-brightness-whole-song-{scheme_id}",
        materials=tuple(materials),
    )
    transformed, role_repairs = _preserve_foreground_roles(score, transformed)
    transformed, chord_repairs = _repair_unsupported(transformed)
    by_id = {material.material_id: material for material in transformed.materials}
    transition_specs = (
        ("transition-ab", "a1-return-prime", "b-theme"),
        ("transition-ba2", "b-return-prime", "a2-theme"),
    )
    paths = []
    for transition_id, source_id, target_id in transition_specs:
        by_id[transition_id], path = _transition_path(
            by_id[transition_id], by_id[source_id], by_id[target_id]
        )
        paths.append(path)
    transformed = replace(
        transformed,
        materials=tuple(by_id[material.material_id] for material in transformed.materials),
    )
    nodes = tuple(
        replace(node, harmonic_focus=9)
        if mode == "major" and node.harmonic_focus is not None
        else node
        for node in plan.nodes
    )
    transformed_plan = replace(
        plan,
        plan_id=f"{plan.plan_id}-brightness-whole-song-{scheme_id}",
        mode=mode,
        nodes=nodes,
    )
    return BrightnessVariant(
        level=level,
        requested=-1 if level == "low" else 1,
        tonal_plan_id=scheme_id,
        plan=transformed_plan,
        score=transformed,
        performance=performance,
        repaired_event_ids=tuple(sorted(set(role_repairs + chord_repairs))),
        transition_paths=tuple(paths),
    )


def build_brightness_variants(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
) -> dict[str, BrightnessVariant]:
    """検証済みの暗い側と明るい側だけを返す。"""
    layout = audit_score_layout(score)
    if (
        len(layout["four_harmony_material_ids"]) != 18
        or layout["two_harmony_material_ids"] != ["transition-ab", "transition-ba2"]
        or layout["harmony_free_material_ids"] != ["ending", "final-tonic"]
    ):
        raise ValueError("brightness v2 input layout does not match v8")
    return {
        "low": _build_variant(
            plan,
            score,
            performance,
            level="low",
            scheme_id="dark-primary",
        ),
        "high": _build_variant(
            plan,
            score,
            performance,
            level="high",
            scheme_id="bright-primary",
        ),
    }


def generated_brightness_descriptor(
    plan: PiecePlan,
    rendered: RenderedPerformance,
) -> dict[str, object]:
    """推定主音ではなくPiecePlanの宣言主音で生成演奏を記述する。"""
    pitch_classes = [0.0] * 12
    for note in rendered.notes:
        pitch_classes[note.pitch % 12] += note.duration_ms
    return {
        "status": "available",
        "tonic_pitch_class": plan.tonal_center,
        "basis": "performed note key duration; pedal extension excluded; occurrences expanded",
        "metrics": fixed_tonic_brightness_metrics(pitch_classes, plan.tonal_center),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def height_descriptor(rendered: RenderedPerformance) -> dict[str, object]:
    """音符イベントで重み付けした高さの交絡診断を返す。"""
    pitches = [note.pitch for note in rendered.notes]
    by_voice: dict[str, list[int]] = defaultdict(list)
    for note in rendered.notes:
        by_voice[note.voice].append(note.pitch)
    return {
        "mean": round(statistics.fmean(pitches), 8),
        "median": round(statistics.median(pitches), 8),
        "p10": round(_percentile(pitches, 0.10), 8),
        "p90": round(_percentile(pitches, 0.90), 8),
        "voice_means": {
            voice: round(statistics.fmean(values), 8) for voice, values in sorted(by_voice.items())
        },
    }


def density_descriptor(rendered: RenderedPerformance) -> dict[str, float]:
    """総尺当たりの音符率と完全一致発音率を旧診断互換で返す。"""
    seconds = rendered.duration_ms / 1000.0
    return {
        "notes_per_second": round(len(rendered.notes) / seconds, 8),
        "attacks_per_second": round(len({note.at_ms for note in rendered.notes}) / seconds, 8),
    }
