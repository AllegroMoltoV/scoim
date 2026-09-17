"""三分曲の全体計画と局所素材を別応答で生成し、決定的に組み立てる。"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import replace
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any, Protocol

from llm_musical_composer.composition_ir import Composition, Material, Note
from llm_musical_composer.music_dsl import (
    THREE_MINUTE_POLICY,
    DslError,
    _arguments,
    _call,
    _list,
    _parse_material,
    _validate,
    parse_composition,
)
from llm_musical_composer.piano_texture import (
    evaluate_piano_texture,
    evaluate_variation_contracts,
)
from llm_musical_composer.run_state import atomic_write_bytes, sha256_json, sha256_text

MAX_MATERIAL_BATCHES = 4
MAX_MATERIALS_PER_BATCH = 8
MIN_MOTIF_MATERIAL_MS = 2_000
MAX_MOTIF_MATERIAL_MS = 6_000
MIN_LONG_FORM_USES = 32
MIN_UNIQUE_MATERIAL_RATIO = 0.60
MAX_MATERIAL_USES_PER_PART = 2
MAX_NATURAL_NOTE_COUNT = 950
NATURAL_DENSITY_SCALE = 0.58
MIN_PHRASE_DURATION_MS = 6_000
MAX_PHRASE_DURATION_MS = 18_000


class LongFormGenerationError(ValueError):
    """段階生成の計画、局所範囲、または組立てが不正な場合の例外。"""


class Runner(Protocol):
    def run(
        self, step_id: str, prompt: str, input_hashes: dict[str, str] | None = None
    ) -> dict[str, object]: ...


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _use_source(use: Any) -> str:
    arguments = [_quoted(use.material_id)]
    if use.role is not None:
        arguments.append(f"role={_quoted(use.role)}")
    if use.energy is not None:
        arguments.append(f"energy={use.energy}")
    if use.attack_style is not None:
        arguments.append(f"attack_style={_quoted(use.attack_style)}")
    return f"use({', '.join(arguments)})"


def _material_source(material: Material) -> str:
    notes = ", ".join(
        "note("
        f"{_quoted(note.event_id)}, at_ms={note.at_ms}, duration_ms={note.duration_ms}, "
        f"pitch={note.pitch}, velocity={note.velocity}"
        f"{f', voice={_quoted(note.voice)}' if note.voice is not None else ''})"
        for note in material.notes
    )
    pedals = ", ".join(
        f"pedal({_quoted(pedal.event_id)}, at_ms={pedal.at_ms}, value={pedal.value})"
        for pedal in material.pedals
    )
    derived_from = (
        f", derived_from={_quoted(material.derived_from)}"
        if material.derived_from is not None
        else ""
    )
    return (
        f"material({_quoted(material.material_id)}, duration_ms={material.duration_ms}, "
        f"notes=[{notes}], pedals=[{pedals}]{derived_from})"
    )


def composition_to_source(composition: Composition) -> str:
    """型付き内部表現を制限付き記法へ決定的に直列化する。"""
    lines = ["composition(", f"    title={_quoted(composition.title)},"]
    if composition.tonal_center is not None:
        lines.append(f"    tonal_center={composition.tonal_center},")
        lines.append(f"    mode={_quoted(composition.mode or '')},")
        assert composition.ending is not None
        lines.append(f"    ending=tonic_hold(duration_ms={composition.ending.duration_ms}),")
    if composition.parts:
        lines.append("    parts=[")
        for part in composition.parts:
            phrases = [phrase for phrase in composition.phrases if phrase.part_id == part.part_id]
            if phrases:
                lines.append(
                    "        part("
                    f"{_quoted(part.part_id)}, role={_quoted(part.role)}, energy={part.energy}, "
                    "phrases=["
                )
                for phrase in phrases:
                    uses = ", ".join(
                        _use_source(use)
                        for use in composition.form[phrase.start_use_index : phrase.end_use_index]
                    )
                    derived_from = (
                        f", derived_from={_quoted(phrase.derived_from)}"
                        if phrase.derived_from is not None
                        else ""
                    )
                    variation_kind = (
                        f", variation_kind={_quoted(phrase.variation_kind)}"
                        if phrase.variation_kind is not None
                        else ""
                    )
                    lines.append(
                        "            phrase("
                        f"{_quoted(phrase.phrase_id)}, role={_quoted(phrase.role)}"
                        f"{derived_from}{variation_kind}, uses=[{uses}]),"
                    )
                lines.append("        ]),")
            else:
                uses = ", ".join(
                    _use_source(use)
                    for use in composition.form[part.start_use_index : part.end_use_index]
                )
                lines.append(
                    "        part("
                    f"{_quoted(part.part_id)}, role={_quoted(part.role)}, energy={part.energy}, "
                    f"uses=[{uses}]),"
                )
        lines.append("    ],")
    else:
        lines.append(f"    form=[{', '.join(_use_source(use) for use in composition.form)}],")
    lines.append("    materials=[")
    lines.extend(f"        {_material_source(material)}," for material in composition.materials)
    lines.extend(["    ],", ")"])
    return "\n".join(lines)


def _parse_material_batch(source: str) -> tuple[str, tuple[Material, ...]]:
    try:
        root = ast.parse(source, mode="eval")
        args = _arguments(
            _call(root.body, "material_batch"),
            positional=("batch_id",),
            required=frozenset({"materials"}),
        )
        batch_id_node = args["batch_id"]
        if not isinstance(batch_id_node, ast.Constant) or not isinstance(batch_id_node.value, str):
            raise DslError("batch_id must be a string literal")
        materials = _list(args["materials"], _parse_material)
        return batch_id_node.value, materials
    except (SyntaxError, DslError) as error:
        raise LongFormGenerationError(f"invalid material batch: {error}") from error


def _normalize_note_bounds(materials: tuple[Material, ...]) -> tuple[Material, ...]:
    """発音しない境界外イベントを除き、素材末尾を越える音価を短縮する。"""
    normalized = []
    for material in materials:
        notes = []
        for note in material.notes:
            if note.at_ms < 0 or note.duration_ms <= 0 or note.at_ms >= material.duration_ms:
                continue
            duration_ms = min(note.duration_ms, material.duration_ms - note.at_ms)
            notes.append(replace(note, duration_ms=duration_ms))
        normalized.append(replace(material, notes=tuple(notes)))
    return tuple(normalized)


def _normalize_same_pitch_retriggers(
    materials: tuple[Material, ...],
) -> tuple[Material, ...]:
    """同時刻の同じ鍵を一音へ縮約し、後続の再打鍵で前音を終える。"""
    normalized: list[Material] = []
    for material in materials:
        best_index_by_key: dict[tuple[int, int], int] = {}
        for index, note in enumerate(material.notes):
            key = (note.pitch, note.at_ms)
            previous_index = best_index_by_key.get(key)
            if previous_index is None:
                best_index_by_key[key] = index
                continue
            previous = material.notes[previous_index]
            if (note.duration_ms, note.velocity) > (previous.duration_ms, previous.velocity):
                best_index_by_key[key] = index
        notes = [
            note
            for index, note in enumerate(material.notes)
            if best_index_by_key[(note.pitch, note.at_ms)] == index
        ]
        indices_by_pitch: dict[int, list[int]] = {}
        for index, note in enumerate(notes):
            indices_by_pitch.setdefault(note.pitch, []).append(index)
        for indices in indices_by_pitch.values():
            ordered = sorted(indices, key=lambda index: (notes[index].at_ms, index))
            for previous_index, current_index in pairwise(ordered):
                previous = notes[previous_index]
                current = notes[current_index]
                if current.at_ms < previous.at_ms + previous.duration_ms:
                    notes[previous_index] = replace(
                        previous, duration_ms=current.at_ms - previous.at_ms
                    )
        normalized.append(replace(material, notes=tuple(notes)))
    return tuple(normalized)


def _normalize_final_pedal_releases(
    materials: tuple[Material, ...],
) -> tuple[Material, ...]:
    """短い素材の余韻を切らないよう、最後の解放だけを素材末尾へ移す。"""
    normalized: list[Material] = []
    for material in materials:
        if not material.pedals:
            normalized.append(material)
            continue
        last_index = max(
            range(len(material.pedals)),
            key=lambda index: (material.pedals[index].at_ms, index),
        )
        last = material.pedals[last_index]
        if last.value != 0:
            normalized.append(material)
            continue
        pedals = list(material.pedals)
        pedals[last_index] = replace(last, at_ms=material.duration_ms)
        normalized.append(replace(material, pedals=tuple(pedals)))
    return tuple(normalized)


def _batches(materials: tuple[Material, ...]) -> list[tuple[Material, ...]]:
    batch_count = min(MAX_MATERIAL_BATCHES, len(materials))
    if not batch_count:
        raise LongFormGenerationError("plan has no materials")
    return [tuple(materials[index::batch_count]) for index in range(batch_count)]


def _batches_for_plan(composition: Composition) -> list[tuple[Material, ...]]:
    """同じ大区分で対比される素材を、可能な限り同じ生成呼び出しへ置く。"""
    if any(material.derived_from is not None for material in composition.materials):
        depth_by_id: dict[str, int] = {}
        for material in composition.materials:
            depth_by_id[material.material_id] = (
                0 if material.derived_from is None else depth_by_id[material.derived_from] + 1
            )
        levels = [
            tuple(
                material
                for material in composition.materials
                if depth_by_id[material.material_id] == depth
            )
            for depth in range(max(depth_by_id.values(), default=0) + 1)
        ]
        batches = [
            level[start : start + MAX_MATERIALS_PER_BATCH]
            for level in levels
            for start in range(0, len(level), MAX_MATERIALS_PER_BATCH)
        ]
        if len(batches) > MAX_MATERIAL_BATCHES:
            raise LongFormGenerationError(
                f"material derivation requires {len(batches)} batches; maximum is "
                f"{MAX_MATERIAL_BATCHES}"
            )
        return batches

    material_order = {
        material.material_id: index for index, material in enumerate(composition.materials)
    }
    components: list[set[str]] = []
    for part in composition.parts:
        current = {
            use.material_id for use in composition.form[part.start_use_index : part.end_use_index]
        }
        overlapping = [component for component in components if component & current]
        if overlapping:
            current.update(*(component for component in overlapping))
            components = [component for component in components if component not in overlapping]
        components.append(current)
    components.sort(key=lambda item: min(material_order[name] for name in item))
    while len(components) > MAX_MATERIAL_BATCHES:
        smallest_index = min(
            range(len(components)),
            key=lambda index: (len(components[index]), index),
        )
        smallest = components.pop(smallest_index)
        destination = 0 if smallest_index != 0 else 1
        components[destination].update(smallest)
        components.sort(key=lambda item: min(material_order[name] for name in item))
    by_id = composition.material_by_id
    return [
        tuple(by_id[name] for name in sorted(component, key=material_order.__getitem__))
        for component in components
    ]


def _validate_long_form_inner_structure(composition: Composition) -> None:
    """長い一素材を大区分と呼ぶ縮退を、音符生成前に拒否する。"""
    if len(composition.form) < MIN_LONG_FORM_USES:
        raise LongFormGenerationError("long-form plan must contain at least 32 uses")
    invalid_duration = [
        material.material_id
        for material in composition.materials
        if not MIN_MOTIF_MATERIAL_MS <= material.duration_ms <= MAX_MOTIF_MATERIAL_MS
    ]
    if invalid_duration:
        raise LongFormGenerationError(
            "long-form motif materials must be between 2000 and 6000 ms: "
            + ", ".join(invalid_duration)
        )
    internal_returns = 0
    for part in composition.parts:
        uses = composition.form[part.start_use_index : part.end_use_index]
        if part.role in {"development", "climax"} and len(uses) < 6:
            raise LongFormGenerationError(
                f"{part.role} part {part.part_id} must contain at least six uses"
            )
        if len({use.material_id for use in uses}) < len(uses):
            internal_returns += 1
    if composition.phrases:
        _validate_long_form_phrase_structure(composition)
    elif internal_returns < 3:
        raise LongFormGenerationError(
            "at least three parts must contain an internal material return"
        )


def _validate_long_form_phrase_structure(composition: Composition) -> None:
    """大区分内の短い変奏と、対比を挟んだ回帰を検査する。"""
    if not composition.phrases:
        raise LongFormGenerationError("long-form phrase structure is required")
    material_map = composition.material_by_id
    phrases_by_part: dict[str, list[Any]] = {part.part_id: [] for part in composition.parts}
    for phrase in composition.phrases:
        phrases_by_part[phrase.part_id].append(phrase)
        duration_ms = sum(
            material_map[use.material_id].duration_ms
            for use in composition.form[phrase.start_use_index : phrase.end_use_index]
        )
        if not MIN_PHRASE_DURATION_MS <= duration_ms <= MAX_PHRASE_DURATION_MS:
            raise LongFormGenerationError(
                f"phrase {phrase.phrase_id} duration must be between "
                f"{MIN_PHRASE_DURATION_MS} and {MAX_PHRASE_DURATION_MS} ms"
            )
    if any(len(phrases) < 2 for phrases in phrases_by_part.values()):
        raise LongFormGenerationError("each long-form part must contain at least two phrases")
    if not any(phrase.role == "variation" for phrase in composition.phrases):
        raise LongFormGenerationError("long-form plan must contain a phrase variation")
    phrase_positions = {phrase.phrase_id: index for index, phrase in enumerate(composition.phrases)}
    has_contrast_return = False
    for index, phrase in enumerate(composition.phrases):
        if phrase.role != "return" or phrase.derived_from is None:
            continue
        source_index = phrase_positions[phrase.derived_from]
        if any(
            intermediate.role == "contrast"
            for intermediate in composition.phrases[source_index + 1 : index]
        ):
            has_contrast_return = True
            break
    if not has_contrast_return:
        raise LongFormGenerationError(
            "long-form plan must contain a return after an intervening contrast phrase"
        )


def _validate_natural_long_form_plan(composition: Composition) -> None:
    """過剰反復を避け、各大区分の末尾に専用フィルインを要求する。"""
    material_ids = [use.material_id for use in composition.form]
    unique_ratio = len(set(material_ids)) / len(material_ids)
    if unique_ratio < MIN_UNIQUE_MATERIAL_RATIO:
        raise LongFormGenerationError(
            f"unique material ratio must be at least {MIN_UNIQUE_MATERIAL_RATIO:.2f}"
        )
    transition_materials: list[str] = []
    for part_index, part in enumerate(composition.parts):
        uses = composition.form[part.start_use_index : part.end_use_index]
        counts: dict[str, int] = {}
        for use in uses:
            counts[use.material_id] = counts.get(use.material_id, 0) + 1
        if max(counts.values(), default=0) > MAX_MATERIAL_USES_PER_PART:
            raise LongFormGenerationError("a material may appear at most twice per part")
        transitions = [index for index, use in enumerate(uses) if use.role == "transition"]
        if part_index < len(composition.parts) - 1:
            if transitions != [len(uses) - 1]:
                raise LongFormGenerationError(
                    "each non-final part must end with one transition use"
                )
            transition_materials.append(uses[-1].material_id)
        elif transitions:
            raise LongFormGenerationError("the final part must not contain a transition use")
    global_counts: dict[str, int] = {}
    for material_id in material_ids:
        global_counts[material_id] = global_counts.get(material_id, 0) + 1
    if any(global_counts[item] != 1 for item in transition_materials):
        raise LongFormGenerationError("transition materials must be unique and used once")
    if any(left == right for left, right in pairwise(material_ids)):
        raise LongFormGenerationError("adjacent uses must not repeat the same material")


def _scale_pitch_classes(tonal_center: int, mode: str) -> tuple[set[int], int]:
    offsets = {0, 2, 4, 5, 7, 9, 11} if mode == "major" else {0, 2, 3, 5, 7, 8, 10}
    scale = {(tonal_center + offset) % 12 for offset in offsets}
    leading_tone = (tonal_center + 11) % 12
    return scale, leading_tone


def _normalize_out_of_scale_notes(
    composition: Composition, materials: tuple[Material, ...]
) -> tuple[Material, ...]:
    """計画した調性で許可されない音符を決定的に除外する。"""
    if composition.tonal_center is None or composition.mode is None:
        raise LongFormGenerationError("pitch normalization requires tonal context")
    scale, leading_tone = _scale_pitch_classes(composition.tonal_center, composition.mode)
    roles_by_material: dict[str, set[str]] = {}
    for use in composition.form:
        if use.role is not None:
            roles_by_material.setdefault(use.material_id, set()).add(use.role)
    normalized = []
    for material in materials:
        allowed = set(scale)
        if roles_by_material.get(material.material_id, set()) & {"transition", "climax"}:
            allowed.add(leading_tone)
        normalized.append(
            replace(
                material,
                notes=tuple(note for note in material.notes if note.pitch % 12 in allowed),
            )
        )
    return tuple(normalized)


def _pitches_form_forbidden_chord(pitches: list[int]) -> bool:
    if len(pitches) > 3:
        return True
    ordered = sorted(pitches)
    return any(
        (right - left) % 12 in {1, 2, 6, 10, 11}
        for left_index, left in enumerate(ordered)
        for right in ordered[left_index + 1 :]
    )


def _normalize_forbidden_ending_chords(
    composition: Composition, materials: tuple[Material, ...]
) -> tuple[Material, ...]:
    """禁止和音を協和部分集合へ縮約し、遷移末尾では導音を優先する。"""
    if composition.tonal_center is None or composition.mode is None:
        raise LongFormGenerationError("transition normalization requires tonal context")
    _, leading_tone = _scale_pitch_classes(composition.tonal_center, composition.mode)
    transition_ids = {use.material_id for use in composition.form if use.role == "transition"}
    normalized: list[Material] = []
    for material in materials:
        indices_by_onset: dict[int, list[int]] = {}
        for index, note in enumerate(material.notes):
            indices_by_onset.setdefault(note.at_ms, []).append(index)
        removed: set[int] = set()
        for indices in indices_by_onset.values():
            pitches = [material.notes[index].pitch for index in indices]
            if not _pitches_form_forbidden_chord(pitches):
                continue
            is_transition_ending = (
                material.material_id in transition_ids
                and material.notes[indices[0]].at_ms >= material.duration_ms - 500
            )
            if is_transition_ending:
                leading = [
                    index for index in indices if material.notes[index].pitch % 12 == leading_tone
                ]
                keep = max(leading or indices, key=lambda index: material.notes[index].pitch)
                removed.update(index for index in indices if index != keep)
                continue
            candidates: list[tuple[int, ...]] = []
            for size in range(min(3, len(indices)), 0, -1):
                candidates = [
                    candidate
                    for candidate in combinations(indices, size)
                    if not _pitches_form_forbidden_chord(
                        [material.notes[index].pitch for index in candidate]
                    )
                ]
                if candidates:
                    break
            keep_indices = max(
                candidates,
                key=lambda candidate: (
                    sum(material.notes[index].velocity for index in candidate),
                    tuple(material.notes[index].pitch for index in candidate),
                ),
            )
            removed.update(index for index in indices if index not in keep_indices)
        normalized.append(
            replace(
                material,
                notes=tuple(
                    note for index, note in enumerate(material.notes) if index not in removed
                ),
            )
        )
    return tuple(normalized)


def _validate_natural_materials(composition: Composition, materials: tuple[Material, ...]) -> None:
    """調性外の音と密集した不協和和音を公開前に拒否する。"""
    if composition.tonal_center is None or composition.mode is None:
        raise LongFormGenerationError("natural harmony validation requires tonal context")
    scale, leading_tone = _scale_pitch_classes(composition.tonal_center, composition.mode)
    roles_by_material: dict[str, set[str]] = {}
    for use in composition.form:
        if use.role is not None:
            roles_by_material.setdefault(use.material_id, set()).add(use.role)
    for material in materials:
        roles = roles_by_material.get(material.material_id, set())
        allowed = set(scale)
        if roles & {"transition", "climax"}:
            allowed.add(leading_tone)
        invalid = sorted({note.pitch % 12 for note in material.notes} - allowed)
        if invalid:
            raise LongFormGenerationError(
                f"material {material.material_id} has pitch classes outside the allowed scale: "
                + ", ".join(map(str, invalid))
            )
        notes_by_onset: dict[int, list[int]] = {}
        for note in material.notes:
            notes_by_onset.setdefault(note.at_ms, []).append(note.pitch)
        for pitches in notes_by_onset.values():
            if len(pitches) > 3:
                raise LongFormGenerationError(
                    f"material {material.material_id} has more than three simultaneous notes"
                )
            ordered = sorted(pitches)
            for left_index, left in enumerate(ordered):
                for right in ordered[left_index + 1 :]:
                    if (right - left) % 12 in {1, 2, 6, 10, 11}:
                        raise LongFormGenerationError(
                            f"material {material.material_id} has a forbidden simultaneous interval"
                        )


def _level(prompt_target: dict[str, Any], axis: str) -> dict[str, float]:
    try:
        level = prompt_target["axis_targets"][axis]["level"]
        return {name: float(level[name]) for name in ("range_low", "median", "range_high")}
    except (KeyError, TypeError, ValueError) as error:
        raise LongFormGenerationError(f"invalid {axis} style target") from error


def _generation_targets(
    composition: Composition,
    prompt_target: dict[str, Any],
    *,
    density_scale: float = 1.0,
) -> dict[str, dict[str, Any]]:
    density = _level(prompt_target, "density")
    polyphony = _level(prompt_target, "polyphony")
    velocity = _level(prompt_target, "velocity")
    register = _level(prompt_target, "register")
    energy_multiplier = {1: 0.55, 2: 0.72, 3: 0.9, 4: 1.08, 5: 1.32}
    uses_by_material: dict[str, list[tuple[int, str, int]]] = {
        material.material_id: [] for material in composition.materials
    }
    for part_index, part in enumerate(composition.parts):
        for use in composition.form[part.start_use_index : part.end_use_index]:
            assert use.energy is not None
            uses_by_material[use.material_id].append((part_index, part.role, use.energy))
    expanded_weights = {
        material.material_id: material.duration_ms * len(uses_by_material[material.material_id])
        for material in composition.materials
    }
    raw_multiplier = {
        material_id: energy_multiplier[max(item[2] for item in usages)]
        for material_id, usages in uses_by_material.items()
    }
    total_weight = sum(expanded_weights.values())
    normalization = (
        sum(expanded_weights[key] * raw_multiplier[key] for key in expanded_weights) / total_weight
    )
    density_center = (density["range_low"] + density["median"]) / 2
    role_register_offset = {
        "opening": 0.0,
        "climax": 7.0,
        "return": 0.0,
        "release": -7.0,
    }
    result: dict[str, dict[str, Any]] = {}
    for material in composition.materials:
        usages = uses_by_material[material.material_id]
        part_index, role, energy = usages[0]
        local_density = (
            density_center * raw_multiplier[material.material_id] / normalization * density_scale
        )
        note_center = max(1, round(local_density * material.duration_ms / 1_000))
        register_offset = role_register_offset.get(role, 5.0 if part_index % 2 == 0 else -5.0)
        polyphony_center = polyphony["median"] * energy_multiplier[energy]
        velocity_center = max(1.0, min(126.0, velocity["median"] + (energy - 3) * 12.0))
        result[material.material_id] = {
            "target_note_count": {
                "low": max(1, round(note_center * 0.9)),
                "center": note_center,
                "high": max(1, round(note_center * 1.1)),
            },
            "target_velocity_median": {
                "low": max(1.0, velocity_center - 6.0),
                "center": velocity_center,
                "high": min(127.0, velocity_center + 6.0),
            },
            "target_pitch_median": {
                "low": register["median"] + register_offset - 3.0,
                "center": register["median"] + register_offset,
                "high": register["median"] + register_offset + 3.0,
            },
            "target_polyphony_mean": {
                "low": max(1.0, polyphony_center - 0.15),
                "center": polyphony_center,
                "high": polyphony_center + 0.15,
            },
        }
    return result


def _material_specs(
    composition: Composition,
    materials: tuple[Material, ...],
    prompt_target: dict[str, Any] | None = None,
    *,
    density_scale: float = 1.0,
    natural_harmony: bool = False,
    accepted_materials: dict[str, Material] | None = None,
) -> list[dict[str, Any]]:
    accepted_materials = accepted_materials or {}
    contexts: dict[str, list[dict[str, Any]]] = {material.material_id: [] for material in materials}
    phrase_by_use_index = {
        use_index: phrase
        for phrase in composition.phrases
        for use_index in range(phrase.start_use_index, phrase.end_use_index)
    }
    for part in composition.parts:
        for use_index in range(part.start_use_index, part.end_use_index):
            use = composition.form[use_index]
            if use.material_id not in contexts:
                continue
            phrase = phrase_by_use_index.get(use_index)
            contexts[use.material_id].append(
                {
                    "part_id": part.part_id,
                    "part_role": part.role,
                    "part_energy": part.energy,
                    "use_role": use.role,
                    "use_energy": use.energy,
                    **(
                        {
                            "phrase_id": phrase.phrase_id,
                            "phrase_role": phrase.role,
                            "phrase_derived_from": phrase.derived_from,
                        }
                        if phrase is not None
                        else {}
                    ),
                }
            )
    generation_targets = (
        _generation_targets(composition, prompt_target, density_scale=density_scale)
        if prompt_target is not None
        else {}
    )
    scale_context: dict[str, Any] = {}
    if natural_harmony:
        if composition.tonal_center is None or composition.mode is None:
            raise LongFormGenerationError("natural harmony generation requires tonal context")
        scale, leading_tone = _scale_pitch_classes(composition.tonal_center, composition.mode)
        scale_context = {
            "tonal_center_pitch_class": composition.tonal_center,
            "mode": composition.mode,
            "allowed_pitch_classes": sorted(scale),
            "leading_tone_pitch_class_for_transition_or_climax_only": leading_tone,
            "maximum_simultaneous_notes": 3,
            "forbidden_simultaneous_interval_classes": [1, 2, 6, 10, 11],
        }
    next_part_by_id = {
        part.part_id: composition.parts[index + 1]
        for index, part in enumerate(composition.parts[:-1])
    }
    transition_targets: dict[str, dict[str, Any]] = {}
    if natural_harmony:
        for material_id, material_contexts in contexts.items():
            for context in material_contexts:
                if context["use_role"] != "transition":
                    continue
                next_part = next_part_by_id[context["part_id"]]
                transition_targets[material_id] = {
                    "part_id": next_part.part_id,
                    "part_role": next_part.role,
                    "part_energy": next_part.energy,
                }
    variation_kind_by_material = {
        composition.form[use_index].material_id: phrase.variation_kind
        for phrase in composition.phrases
        if phrase.role == "variation" and phrase.variation_kind is not None
        for use_index in range(phrase.start_use_index, phrase.end_use_index)
        if composition.form[use_index].role != "transition"
    }
    return [
        {
            "material_id": material.material_id,
            "duration_ms": material.duration_ms,
            "usages": contexts[material.material_id],
            "voice_contract": {
                "required_voices": ["upper", "lower"],
                "onset_tolerance_ms": 50,
                "avoid_lockstep": True,
                "avoid_temporal_separation": True,
            },
            **(
                {"variation_kind": variation_kind_by_material[material.material_id]}
                if material.material_id in variation_kind_by_material
                else {}
            ),
            **(
                {"generation_targets": generation_targets[material.material_id]}
                if prompt_target is not None
                else {}
            ),
            **({"tonal_context": scale_context} if natural_harmony else {}),
            **(
                {"transition_to": transition_targets[material.material_id]}
                if material.material_id in transition_targets
                else {}
            ),
            **(
                {"source_material": _material_source(accepted_materials[material.derived_from])}
                if material.derived_from is not None and material.derived_from in accepted_materials
                else {}
            ),
        }
        for material in materials
    ]


def _material_musical_signature(material: Material) -> tuple[Any, ...]:
    return (
        material.duration_ms,
        tuple(
            (note.at_ms, note.duration_ms, note.pitch, note.velocity, note.voice)
            for note in material.notes
        ),
        tuple((pedal.at_ms, pedal.value) for pedal in material.pedals),
    )


def _validate_material_variations(
    materials: tuple[Material, ...],
    accepted_materials: dict[str, Material],
    *,
    allow_exact_copy: bool = False,
) -> None:
    for material in materials:
        if material.derived_from is None:
            continue
        source = accepted_materials.get(material.derived_from)
        if source is None:
            raise LongFormGenerationError(
                f"source material {material.derived_from} was not accepted before variation "
                f"{material.material_id}"
            )
        if not allow_exact_copy and _material_musical_signature(
            material
        ) == _material_musical_signature(source):
            raise LongFormGenerationError(
                f"variation material {material.material_id} is an exact copy of "
                f"{material.derived_from}"
            )


def _validate_variation_kinds(composition: Composition) -> None:
    for phrase in composition.phrases:
        if phrase.role == "variation" and phrase.variation_kind is None:
            raise LongFormGenerationError(
                f"variation phrase {phrase.phrase_id} requires variation_kind"
            )


def _validate_voice_and_variations(composition: Composition) -> None:
    """長尺版で必須となる二声性と型付き変奏を一緒に検査する。"""
    _validate_variation_kinds(composition)
    texture = evaluate_piano_texture(composition)
    if texture["status"] != "pass":
        raise LongFormGenerationError(str(texture["issues"][0]))
    variations = evaluate_variation_contracts(composition)
    if variations["status"] != "pass":
        raise LongFormGenerationError(str(variations["issues"][0]))


def _warp_rhythmic_material(material: Material) -> Material:
    """上声の音高列を保ったまま、発音位置へ単調なリズム変形を加える。"""
    onsets = sorted({note.at_ms for note in material.notes if note.voice == "upper"})
    if len(onsets) < 3 or onsets[-1] <= onsets[0]:
        return material
    first, last = onsets[0], onsets[-1]
    span = last - first
    mapped = {first: first, last: last}
    previous = first
    for index, onset in enumerate(onsets[1:-1], 1):
        normalized = (onset - first) / span
        warped = round(first + span * normalized**1.25)
        remaining = len(onsets) - index - 1
        warped = max(previous + 1, min(warped, last - remaining))
        mapped[onset] = warped
        previous = warped
    notes = tuple(
        replace(note, at_ms=mapped[note.at_ms])
        if note.voice == "upper" and note.at_ms in mapped
        else note
        for note in material.notes
    )
    return _normalize_same_pitch_retriggers((replace(material, notes=notes),))[0]


def _repair_exact_material_copies(
    accepted_materials: dict[str, Material],
) -> dict[str, Material]:
    """派生宣言と矛盾する完全コピーだけを決定的なリズム差へ変える。"""
    updated = dict(accepted_materials)
    for material_id, material in accepted_materials.items():
        if material.derived_from is None:
            continue
        source = accepted_materials[material.derived_from]
        if _material_musical_signature(material) != _material_musical_signature(source):
            continue
        updated[material_id] = _warp_rhythmic_material(material)
    return updated


def _restore_rhythmic_contour(source: Material, variation: Material) -> Material:
    """リズム変奏の演奏情報を保ち、上声音高列だけを派生元へ戻す。"""
    source_groups: dict[int, list[Note]] = {}
    variation_groups: dict[int, list[Note]] = {}
    for note in source.notes:
        if note.voice == "upper":
            source_groups.setdefault(note.at_ms, []).append(note)
    for note in variation.notes:
        if note.voice == "upper":
            variation_groups.setdefault(note.at_ms, []).append(note)
    source_top = [
        max(source_groups[onset], key=lambda note: (note.pitch, note.event_id))
        for onset in sorted(source_groups)
    ]
    if len(source_top) < 3 or len(variation_groups) < 3:
        return variation
    pitch_by_event_id: dict[str, int] = {}
    for index, onset in enumerate(sorted(variation_groups)):
        group = variation_groups[onset]
        original_top = max(group, key=lambda note: (note.pitch, note.event_id))
        target_top = source_top[index % len(source_top)].pitch
        pitch_by_event_id[original_top.event_id] = target_top
        used = {target_top}
        for note in sorted(group, key=lambda item: (item.pitch, item.event_id), reverse=True):
            if note.event_id == original_top.event_id:
                continue
            pitch = note.pitch
            while pitch >= target_top or pitch in used:
                pitch -= 12
            if pitch < 21:
                return variation
            pitch_by_event_id[note.event_id] = pitch
            used.add(pitch)
    notes = tuple(
        replace(note, pitch=pitch_by_event_id[note.event_id])
        if note.event_id in pitch_by_event_id
        else note
        for note in variation.notes
    )
    return _normalize_same_pitch_retriggers((replace(variation, notes=notes),))[0]


def _restore_textural_upper_motif(source: Material, variation: Material) -> Material:
    """テクスチャ変奏の下声を保ち、上声主題だけを派生元へ戻す。"""
    source_upper = sorted(
        (note for note in source.notes if note.voice == "upper"),
        key=lambda note: (note.at_ms, note.pitch, note.duration_ms, note.event_id),
    )
    if len(source_upper) < 3:
        return variation
    scale = variation.duration_ms / source.duration_ms
    restored_upper = tuple(
        replace(
            note,
            event_id=f"{variation.material_id}-texture-u-{index:03d}",
            at_ms=at_ms,
            duration_ms=min(max(1, round(note.duration_ms * scale)), variation.duration_ms - at_ms),
        )
        for index, note in enumerate(source_upper, 1)
        for at_ms in (round(note.at_ms * scale),)
    )
    other_notes = tuple(note for note in variation.notes if note.voice != "upper")
    notes = tuple(
        sorted(
            (*other_notes, *restored_upper),
            key=lambda note: (note.at_ms, note.pitch, note.event_id),
        )
    )
    return _normalize_same_pitch_retriggers((replace(variation, notes=notes),))[0]


def _restore_registral_pattern(
    source: Material,
    variation: Material,
    *,
    allowed_pitch_classes: frozenset[int] | None = None,
) -> Material:
    """十分な対応音から移調量を推定し、欠けた上声音を決定的に復元する。"""
    source_upper = sorted(
        (note for note in source.notes if note.voice == "upper"),
        key=lambda note: (note.at_ms, note.duration_ms, note.pitch, note.event_id),
    )
    variation_upper = sorted(
        (note for note in variation.notes if note.voice == "upper"),
        key=lambda note: (note.at_ms, note.duration_ms, note.pitch, note.event_id),
    )
    if len(source_upper) < 3 or len(variation_upper) < 2:
        return variation
    source_keys = [(note.at_ms, note.duration_ms) for note in source_upper]
    variation_keys = [(note.at_ms, note.duration_ms) for note in variation_upper]
    if len(set(source_keys)) != len(source_keys) or len(set(variation_keys)) != len(variation_keys):
        return variation
    source_by_key = dict(zip(source_keys, source_upper, strict=True))
    variation_by_key = dict(zip(variation_keys, variation_upper, strict=True))
    shifts = [
        variation_by_key[key].pitch - source_note.pitch
        for key, source_note in source_by_key.items()
        if key in variation_by_key
    ]
    if len(shifts) < 2:
        return variation
    shift, support = Counter(shifts).most_common(1)[0]
    if abs(shift) < 5 or support / len(shifts) < 0.8:
        return variation
    restored_upper = []
    for index, source_note in enumerate(source_upper, 1):
        existing = variation_by_key.get((source_note.at_ms, source_note.duration_ms))
        pitch = existing.pitch if existing is not None else source_note.pitch + shift
        if allowed_pitch_classes is not None and pitch % 12 not in allowed_pitch_classes:
            alternatives = [
                candidate
                for distance in (1, 2)
                for candidate in (pitch - distance, pitch + distance)
                if 21 <= candidate <= 108 and candidate % 12 in allowed_pitch_classes
            ]
            if not alternatives:
                return variation
            pitch = alternatives[0]
        if not 21 <= pitch <= 108:
            return variation
        restored_upper.append(
            replace(
                source_note,
                event_id=(
                    existing.event_id
                    if existing is not None
                    else f"{variation.material_id}-reg-{index:03d}"
                ),
                pitch=pitch,
                velocity=existing.velocity if existing is not None else source_note.velocity,
            )
        )
    lower_and_untyped = [note for note in variation.notes if note.voice != "upper"]
    notes = tuple(
        sorted(
            (*lower_and_untyped, *restored_upper),
            key=lambda note: (note.at_ms, note.pitch, note.event_id),
        )
    )
    return replace(variation, notes=notes)


def transpose_composition(composition: Composition, semitones: int) -> Composition:
    """構成と演奏情報を保ったまま、全音符と調性中心を一律移調する。"""
    if isinstance(semitones, bool) or not isinstance(semitones, int):
        raise TypeError("semitones must be an integer")
    if composition.tonal_center is None:
        raise LongFormGenerationError("transposition requires a tonal center")
    materials = []
    for material in composition.materials:
        notes = []
        for note in material.notes:
            pitch = note.pitch + semitones
            if not 21 <= pitch <= 108:
                raise LongFormGenerationError("transposition moves a note outside the piano range")
            notes.append(replace(note, pitch=pitch))
        materials.append(replace(material, notes=tuple(notes)))
    return replace(
        composition,
        tonal_center=(composition.tonal_center + semitones) % 12,
        materials=tuple(materials),
    )


def _repair_failed_variations(
    runner: Runner,
    composition: Composition,
    accepted_materials: dict[str, Material],
    *,
    material_prompt_template: str,
    base_input_hashes: dict[str, str],
    natural_harmony: bool,
) -> dict[str, Material]:
    """不合格の変奏素材だけを一度再生成し、他の確定素材を保持する。"""
    report = evaluate_variation_contracts(composition)
    failed_reports = [pair for pair in report["pairs"] if pair["status"] != "pass"]
    if not failed_reports:
        return dict(accepted_materials)
    failed_ids = {str(pair["variation_material_id"]) for pair in failed_reports}
    expected = tuple(
        material for material in composition.materials if material.material_id in failed_ids
    )
    if set(failed_ids) != {material.material_id for material in expected}:
        raise LongFormGenerationError("failed variation material is absent from the plan")
    issues_by_id = {
        str(pair["variation_material_id"]): list(pair["issues"]) for pair in failed_reports
    }
    specs = _material_specs(
        composition,
        expected,
        natural_harmony=natural_harmony,
        accepted_materials=accepted_materials,
    )
    for spec in specs:
        material_id = str(spec["material_id"])
        spec["repair_target"] = _material_source(accepted_materials[material_id])
        spec["validation_issues"] = issues_by_id[material_id]
    batch_id = "repair-variations"
    prompt = _render_prompt(
        material_prompt_template,
        {
            "batch_id": batch_id,
            "material_specs": json.dumps(specs, ensure_ascii=False, sort_keys=True),
        },
    )
    response = runner.run(
        "material-repair-variations",
        prompt,
        {
            **base_input_hashes,
            "repair_spec": sha256_json(specs),
            "invalid_variations": sha256_json(
                {
                    material_id: _material_source(accepted_materials[material_id])
                    for material_id in failed_ids
                }
            ),
        },
    )
    returned_batch_id, repaired = _parse_material_batch(_response_source(response))
    if returned_batch_id != batch_id:
        raise LongFormGenerationError("variation repair batch ID changed")
    repaired = _normalize_note_bounds(repaired)
    repaired = _normalize_same_pitch_retriggers(repaired)
    if natural_harmony:
        repaired = _normalize_out_of_scale_notes(composition, repaired)
        repaired = _normalize_forbidden_ending_chords(composition, repaired)
        _validate_natural_materials(composition, repaired)
        repaired = _normalize_final_pedal_releases(repaired)
    returned_map = {material.material_id: material for material in repaired}
    if len(returned_map) != len(repaired) or set(returned_map) != failed_ids:
        raise LongFormGenerationError("variation repair IDs do not exactly match failed materials")
    expected_contract = {
        material.material_id: (material.duration_ms, material.derived_from) for material in expected
    }
    returned_contract = {
        material.material_id: (material.duration_ms, material.derived_from) for material in repaired
    }
    if returned_contract != expected_contract:
        raise LongFormGenerationError("variation repair changed duration or derivation")
    _validate_material_variations(repaired, accepted_materials)
    updated = dict(accepted_materials)
    updated.update(returned_map)
    repaired_composition = replace(
        composition,
        materials=tuple(updated[material.material_id] for material in composition.materials),
    )
    repaired_report = evaluate_variation_contracts(repaired_composition)
    for pair in repaired_report["pairs"]:
        if pair["status"] == "pass":
            continue
        material_id = str(pair["variation_material_id"])
        if pair["variation_kind"] == "rhythmic":
            source_id = str(pair["source_material_id"])
            if "rhythmic variation must preserve the upper contour" in pair["issues"]:
                updated[material_id] = _restore_rhythmic_contour(
                    updated[source_id], updated[material_id]
                )
            if "rhythmic variation must audibly change rhythm" in pair["issues"]:
                updated[material_id] = _warp_rhythmic_material(updated[material_id])
        elif pair["variation_kind"] == "textural" and (
            "textural variation must preserve the upper motif" in pair["issues"]
        ):
            source_id = str(pair["source_material_id"])
            updated[material_id] = _restore_textural_upper_motif(
                updated[source_id], updated[material_id]
            )
        elif pair["variation_kind"] == "registral" and pair["issues"] == [
            "registral variation must preserve rhythm and contour"
        ]:
            source_id = str(pair["source_material_id"])
            allowed_pitch_classes = None
            if natural_harmony:
                assert composition.tonal_center is not None
                assert composition.mode is not None
                scale, leading_tone = _scale_pitch_classes(
                    composition.tonal_center, composition.mode
                )
                allowed = set(scale)
                roles = {
                    use.role
                    for use in composition.form
                    if use.material_id == material_id and use.role is not None
                }
                if roles & {"transition", "climax"}:
                    allowed.add(leading_tone)
                allowed_pitch_classes = frozenset(allowed)
            updated[material_id] = _restore_registral_pattern(
                updated[source_id],
                updated[material_id],
                allowed_pitch_classes=allowed_pitch_classes,
            )
    repaired_composition = replace(
        composition,
        materials=tuple(updated[material.material_id] for material in composition.materials),
    )
    if natural_harmony:
        _validate_natural_materials(
            repaired_composition, tuple(updated[material_id] for material_id in failed_ids)
        )
    _validate_voice_and_variations(repaired_composition)
    return updated


def _render_prompt(template: str, replacements: dict[str, str]) -> str:
    result = template
    for name, value in replacements.items():
        result = result.replace("{{" + name + "}}", value)
    if re.search(r"\{\{[A-Za-z0-9_]+\}\}", result):
        raise LongFormGenerationError("material prompt contains unresolved variables")
    return result


def _response_source(response: dict[str, object]) -> str:
    source = response.get("composition_source")
    if not isinstance(source, str) or not source:
        raise LongFormGenerationError("response has no composition_source")
    return source


def run_staged_generation(
    runner: Runner,
    *,
    plan_prompt: str,
    material_prompt_template: str,
    output_dir: Path,
    base_input_hashes: dict[str, str] | None = None,
    style_prompt_target: dict[str, Any] | None = None,
    require_inner_structure: bool = False,
    require_naturalness: bool = False,
    require_phrase_structure: bool = False,
    require_voice_structure: bool = False,
) -> Composition:
    """全体計画を固定し、局所素材だけを別応答から組み立てる。"""
    output_dir = Path(output_dir)
    base_hashes = dict(base_input_hashes or {})
    plan_source = _response_source(runner.run("plan", plan_prompt, base_hashes))
    try:
        plan = parse_composition(
            plan_source,
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )
    except DslError as error:
        raise LongFormGenerationError(f"invalid long-form plan: {error}") from error
    if any(material.notes or material.pedals for material in plan.materials):
        raise LongFormGenerationError("long-form plan must not contain notes or pedals")
    if require_inner_structure:
        _validate_long_form_inner_structure(plan)
    if require_phrase_structure:
        _validate_long_form_phrase_structure(plan)
    if require_voice_structure:
        _validate_variation_kinds(plan)
    if require_naturalness:
        _validate_natural_long_form_plan(plan)

    accepted: dict[str, Material] = {}
    plan_hash = sha256_text(plan_source)
    has_derivations = any(material.derived_from is not None for material in plan.materials)
    batches = (
        _batches_for_plan(plan)
        if require_inner_structure or has_derivations
        else _batches(plan.materials)
    )
    for index, expected in enumerate(batches, 1):
        batch_id = f"batch-{index:02d}"
        specs = _material_specs(
            plan,
            expected,
            style_prompt_target,
            density_scale=NATURAL_DENSITY_SCALE if require_naturalness else 1.0,
            natural_harmony=require_naturalness,
            accepted_materials=accepted,
        )
        dependencies = {
            material.derived_from: _material_source(accepted[material.derived_from])
            for material in expected
            if material.derived_from is not None
        }
        prompt = _render_prompt(
            material_prompt_template,
            {
                "batch_id": batch_id,
                "material_specs": json.dumps(specs, ensure_ascii=False, sort_keys=True),
            },
        )
        response = runner.run(
            f"material-{batch_id}",
            prompt,
            {
                **base_hashes,
                "plan": plan_hash,
                "batch_spec": sha256_json(specs),
                **({"dependency_materials": sha256_json(dependencies)} if dependencies else {}),
            },
        )
        returned_batch_id, materials = _parse_material_batch(_response_source(response))
        materials = _normalize_note_bounds(materials)
        materials = _normalize_same_pitch_retriggers(materials)
        if require_naturalness:
            materials = _normalize_out_of_scale_notes(plan, materials)
            materials = _normalize_forbidden_ending_chords(plan, materials)
            _validate_natural_materials(plan, materials)
            materials = _normalize_final_pedal_releases(materials)
        _validate_material_variations(
            materials,
            accepted,
            allow_exact_copy=require_voice_structure,
        )
        if returned_batch_id != batch_id:
            raise LongFormGenerationError("material batch ID changed")
        expected_map = {
            material.material_id: (material.duration_ms, material.derived_from)
            for material in expected
        }
        returned_map = {
            material.material_id: (material.duration_ms, material.derived_from)
            for material in materials
        }
        if len(returned_map) != len(materials) or set(returned_map) != set(expected_map):
            raise LongFormGenerationError("material IDs do not exactly match the batch plan")
        if returned_map != expected_map:
            if any(
                returned_map[material_id][0] != expected_map[material_id][0]
                for material_id in expected_map
            ):
                raise LongFormGenerationError("material duration changed from the plan")
            raise LongFormGenerationError("material derivation changed from the plan")
        accepted.update({material.material_id: material for material in materials})

    if require_voice_structure:
        accepted = _repair_exact_material_copies(accepted)
    final = replace(plan, materials=tuple(accepted[item.material_id] for item in plan.materials))
    if require_voice_structure and evaluate_variation_contracts(final)["status"] != "pass":
        accepted = _repair_failed_variations(
            runner,
            final,
            accepted,
            material_prompt_template=material_prompt_template,
            base_input_hashes={**base_hashes, "plan": plan_hash},
            natural_harmony=require_naturalness,
        )
        final = replace(
            plan,
            materials=tuple(accepted[item.material_id] for item in plan.materials),
        )
    _validate_material_variations(final.materials, accepted)
    if require_naturalness and final.note_count > MAX_NATURAL_NOTE_COUNT:
        raise LongFormGenerationError(
            f"assembled composition exceeds {MAX_NATURAL_NOTE_COUNT} notes"
        )
    try:
        _validate(
            final,
            require_section_contract=True,
            policy=THREE_MINUTE_POLICY,
        )
    except DslError as error:
        raise LongFormGenerationError(f"assembled composition is invalid: {error}") from error
    if require_voice_structure:
        _validate_voice_and_variations(final)
    final_source = composition_to_source(final)
    atomic_write_bytes(output_dir / "plan.music.py", (plan_source + "\n").encode("utf-8"))
    atomic_write_bytes(output_dir / "final.music.py", (final_source + "\n").encode("utf-8"))
    return final
