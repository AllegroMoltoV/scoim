"""対比素材の時間配置だけを変え、3分曲の変化量端点を作る。"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, replace
from itertools import permutations
from typing import Any

from llm_musical_composer.composition_ir import Composition, Material, Note, Pedal
from llm_musical_composer.long_form_generation import (
    _normalize_note_bounds,
    _normalize_same_pitch_retriggers,
)
from llm_musical_composer.song_change import _expand_use_range, _relation, describe_song_change

MAX_ARRANGEMENT_SLOTS = 8


class SongChangeArrangementError(ValueError):
    """安全な低・高配置を作れない場合の例外。"""


@dataclass(frozen=True)
class ArrangementTriplet:
    low: Composition
    center: Composition
    high: Composition
    selection: dict[str, Any]


@dataclass(frozen=True)
class _Slot:
    use_index: int
    phrase_id: str
    target: Material
    source_phrase: Material


def _contrast_slots(composition: Composition) -> tuple[_Slot, ...]:
    phrases = list(composition.phrases)
    candidates_by_phrase: list[list[tuple[int, str]]] = []
    contrast_phrases = []
    for phrase_index, phrase in enumerate(phrases):
        if phrase.role != "contrast":
            continue
        prior_statement = next(
            (
                candidate
                for candidate in reversed(phrases[:phrase_index])
                if candidate.role == "statement"
            ),
            None,
        )
        if prior_statement is None:
            raise SongChangeArrangementError(
                f"contrast phrase {phrase.phrase_id} has no prior statement"
            )
        uses = [
            (use_index, composition.form[use_index].material_id)
            for use_index in range(phrase.start_use_index, phrase.end_use_index)
            if composition.form[use_index].role != "transition"
        ]
        if uses:
            contrast_phrases.append((phrase, prior_statement))
            candidates_by_phrase.append(uses)

    selected: list[tuple[int, int, str]] = []
    depth = 0
    while len(selected) < MAX_ARRANGEMENT_SLOTS:
        added = False
        for phrase_index, candidates in enumerate(candidates_by_phrase):
            if depth >= len(candidates):
                continue
            use_index, material_id = candidates[depth]
            selected.append((phrase_index, use_index, material_id))
            added = True
            if len(selected) == MAX_ARRANGEMENT_SLOTS:
                break
        if not added:
            break
        depth += 1
    if len(selected) < 3:
        raise SongChangeArrangementError("at least three contrast use slots are required")

    slots = []
    for phrase_index, use_index, material_id in selected:
        phrase, source = contrast_phrases[phrase_index]
        slots.append(
            _Slot(
                use_index,
                phrase.phrase_id,
                composition.material_by_id[material_id],
                _expand_use_range(
                    composition,
                    source.phrase_id,
                    source.start_use_index,
                    source.end_use_index,
                ),
            )
        )
    return tuple(slots)


def _scale_material(donor: Material, target: Material, *, material_id: str) -> Material:
    scale = target.duration_ms / donor.duration_ms
    notes = tuple(
        Note(
            event_id=f"{material_id}-n-{index:04d}",
            at_ms=round(note.at_ms * scale),
            duration_ms=max(1, round(note.duration_ms * scale)),
            pitch=note.pitch,
            velocity=note.velocity,
            voice=note.voice,
        )
        for index, note in enumerate(donor.notes, 1)
    )
    pedals = tuple(
        Pedal(
            event_id=f"{material_id}-p-{index:03d}",
            at_ms=round(pedal.at_ms * scale),
            value=pedal.value,
        )
        for index, pedal in enumerate(donor.pedals, 1)
    )
    prepared = Material(material_id, target.duration_ms, notes, pedals)
    bounded = _normalize_note_bounds((prepared,))[0]
    normalized = _normalize_same_pitch_retriggers((bounded,))[0]
    if len(normalized.notes) != len(donor.notes):
        raise SongChangeArrangementError("time scaling changed the selected note count")
    return normalized


def _distance_components(source: Material, target: Material) -> tuple[float, ...]:
    relation, _ = _relation(
        source,
        target,
        relation="contrast",
        source_id="statement",
        target_id=target.material_id,
    )
    names = (
        "upper_interval_similarity",
        "upper_rhythm_similarity",
        "lower_interval_similarity",
        "lower_rhythm_similarity",
        "pitch_class_jaccard",
    )
    if any(relation[name] is None for name in names):
        raise SongChangeArrangementError("contrast relation has unavailable metrics")
    similarities = tuple(1.0 - float(relation[name]) for name in names)
    register = relation["upper_register_shift_semitones"]
    assert register is not None
    return (*similarities, min(float(register) / 12.0, 1.0))


def _assignment(
    slots: tuple[_Slot, ...], donors: tuple[Material, ...], order: tuple[int, ...], label: str
) -> tuple[dict[str, Material], tuple[float, ...]]:
    replacements = {}
    distances = []
    for slot_index, (slot, donor_index) in enumerate(zip(slots, order, strict=True), 1):
        material_id = f"song-change-{label}-{slot_index:02d}"
        material = _scale_material(donors[donor_index], slot.target, material_id=material_id)
        replacements[material_id] = material
        distances.extend(_distance_components(slot.source_phrase, material))
    return replacements, tuple(distances)


def _low_key(distances: tuple[float, ...]) -> tuple[float, float, float]:
    return (max(distances), statistics.median(distances), sum(distances))


def _high_key(distances: tuple[float, ...]) -> tuple[float, float, float]:
    return (min(distances), statistics.median(distances), sum(distances))


def _compose_endpoint(
    composition: Composition,
    slots: tuple[_Slot, ...],
    donors: tuple[Material, ...],
    order: tuple[int, ...],
    label: str,
) -> Composition:
    replacements, _ = _assignment(slots, donors, order, label)
    form = list(composition.form)
    for slot_index, slot in enumerate(slots, 1):
        form[slot.use_index] = replace(
            form[slot.use_index], material_id=f"song-change-{label}-{slot_index:02d}"
        )
    return replace(
        composition,
        form=tuple(form),
        materials=(*composition.materials, *replacements.values()),
    )


def _performance_multiset(composition: Composition) -> tuple[Counter[int], Counter[int]]:
    pitches: Counter[int] = Counter()
    velocities: Counter[int] = Counter()
    materials = composition.material_by_id
    for use in composition.form:
        for note in materials[use.material_id].notes:
            pitches[note.pitch] += 1
            velocities[note.velocity] += 1
    return pitches, velocities


def derive_arrangement_triplet(composition: Composition) -> ArrangementTriplet:
    """同じ素材内容集合から、近い対比と遠い対比の配置を決定的に作る。"""
    slots = _contrast_slots(composition)
    donors = tuple(slot.target for slot in slots)
    identity = tuple(range(len(slots)))
    _, center_distances = _assignment(slots, donors, identity, "center-diagnostic")
    candidates = []
    for order in permutations(range(len(slots))):
        if order == identity:
            continue
        _, distances = _assignment(slots, donors, order, "candidate-diagnostic")
        candidates.append((order, distances))

    low_candidates = sorted(candidates, key=lambda item: (_low_key(item[1]), item[0]))
    high_candidates = sorted(
        candidates,
        key=lambda item: (_high_key(item[1]), item[0]),
        reverse=True,
    )
    low_order, low_distances = low_candidates[0]
    high_choice = next((item for item in high_candidates if item[0] != low_order), None)
    if high_choice is None:
        raise SongChangeArrangementError("low and high assignments cannot be separated")
    high_order, high_distances = high_choice
    if _low_key(low_distances) >= _low_key(center_distances):
        raise SongChangeArrangementError("low assignment is not closer than the center")
    if _high_key(high_distances) <= _high_key(center_distances):
        raise SongChangeArrangementError("high assignment is not farther than the center")

    low = _compose_endpoint(composition, slots, donors, low_order, "low")
    high = _compose_endpoint(composition, slots, donors, high_order, "high")
    center_hold = describe_song_change(composition)["hold"]
    low_hold = describe_song_change(low)["hold"]
    high_hold = describe_song_change(high)["hold"]
    for name, endpoint, hold in (("low", low, low_hold), ("high", high, high_hold)):
        if endpoint.note_count != composition.note_count:
            raise SongChangeArrangementError(f"{name} endpoint changed the note count")
        if _performance_multiset(endpoint) != _performance_multiset(composition):
            raise SongChangeArrangementError(f"{name} endpoint changed pitch or velocity content")
        center_polyphony = float(center_hold["mean_polyphony"])
        if abs(float(hold["mean_polyphony"]) / center_polyphony - 1.0) > 0.05:
            raise SongChangeArrangementError(f"{name} endpoint changed mean polyphony over 5%")

    return ArrangementTriplet(
        low,
        composition,
        high,
        {
            "slot_count": len(slots),
            "use_indices": [slot.use_index for slot in slots],
            "low_order": list(low_order),
            "center_order": list(identity),
            "high_order": list(high_order),
            "low_objective": list(_low_key(low_distances)),
            "center_low_objective": list(_low_key(center_distances)),
            "center_high_objective": list(_high_key(center_distances)),
            "high_objective": list(_high_key(high_distances)),
            "hold": {"center": center_hold, "low": low_hold, "high": high_hold},
        },
    )
