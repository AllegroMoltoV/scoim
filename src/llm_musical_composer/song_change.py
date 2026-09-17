"""曲内の提示、変奏、対比、回帰の実音関係を合算せずに記述する。"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence
from difflib import SequenceMatcher
from itertools import combinations, pairwise
from typing import Any

from llm_musical_composer.composition_ir import Composition, Material, Note


def _round(value: float) -> float:
    return round(float(value), 8)


def _expand_use_range(
    composition: Composition,
    material_id: str,
    start_use_index: int,
    end_use_index: int,
) -> Material:
    materials = composition.material_by_id
    offset = 0
    notes: list[Note] = []
    for use_index, use in enumerate(
        composition.form[start_use_index:end_use_index], start=start_use_index
    ):
        material = materials[use.material_id]
        notes.extend(
            Note(
                event_id=f"use-{use_index}-{note.event_id}",
                at_ms=offset + note.at_ms,
                duration_ms=note.duration_ms,
                pitch=note.pitch,
                velocity=note.velocity,
                voice=note.voice,
            )
            for note in material.notes
        )
        offset += material.duration_ms
    return Material(material_id, offset, tuple(notes))


def _top_line(material: Material, voice: str) -> list[Note]:
    grouped: dict[int, list[Note]] = defaultdict(list)
    for note in material.notes:
        if note.voice == voice:
            grouped[note.at_ms].append(note)
    return [max(grouped[onset], key=lambda note: note.pitch) for onset in sorted(grouped)]


def _intervals(notes: Sequence[Note]) -> tuple[int, ...]:
    return tuple(right.pitch - left.pitch for left, right in pairwise(notes))


def _rhythm(notes: Sequence[Note]) -> tuple[int, ...]:
    if len(notes) < 2:
        return ()
    first = notes[0].at_ms
    span = notes[-1].at_ms - first
    if span <= 0:
        return ()
    return tuple(round((note.at_ms - first) * 16 / span) for note in notes)


def _normalized_positions(notes: Sequence[Note]) -> tuple[float, ...]:
    if len(notes) < 2:
        return ()
    first = notes[0].at_ms
    span = notes[-1].at_ms - first
    if span <= 0:
        return ()
    return tuple((note.at_ms - first) / span for note in notes)


def _mean_distance(first: Sequence[float], second: Sequence[float]) -> float | None:
    if not first or len(first) != len(second):
        return None
    return _round(
        sum(abs(left - right) for left, right in zip(first, second, strict=True)) / len(first)
    )


def _similarity(first: Sequence[int], second: Sequence[int]) -> float | None:
    if not first or not second:
        return None
    return _round(SequenceMatcher(a=first, b=second, autojunk=False).ratio())


def _pitch_class_jaccard(first: Material, second: Material) -> float | None:
    left = {note.pitch % 12 for note in first.notes}
    right = {note.pitch % 12 for note in second.notes}
    if not left or not right:
        return None
    return _round(len(left & right) / len(left | right))


def _relation(
    source: Material,
    target: Material,
    *,
    relation: str,
    source_id: str,
    target_id: str,
    variation_kind: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    source_upper = _top_line(source, "upper")
    target_upper = _top_line(target, "upper")
    source_lower = _top_line(source, "lower")
    target_lower = _top_line(target, "lower")
    metrics: dict[str, float | None] = {
        "upper_interval_similarity": _similarity(
            _intervals(source_upper), _intervals(target_upper)
        ),
        "upper_rhythm_similarity": _similarity(_rhythm(source_upper), _rhythm(target_upper)),
        "upper_rhythm_position_distance": _mean_distance(
            _normalized_positions(source_upper), _normalized_positions(target_upper)
        ),
        "upper_interval_distance_semitones": _mean_distance(
            _intervals(source_upper), _intervals(target_upper)
        ),
        "lower_interval_similarity": _similarity(
            _intervals(source_lower), _intervals(target_lower)
        ),
        "lower_rhythm_similarity": _similarity(_rhythm(source_lower), _rhythm(target_lower)),
        "lower_rhythm_position_distance": _mean_distance(
            _normalized_positions(source_lower), _normalized_positions(target_lower)
        ),
        "lower_interval_distance_semitones": _mean_distance(
            _intervals(source_lower), _intervals(target_lower)
        ),
        "upper_register_shift_semitones": (
            _round(
                abs(
                    statistics.median(note.pitch for note in target_upper)
                    - statistics.median(note.pitch for note in source_upper)
                )
            )
            if source_upper and target_upper
            else None
        ),
        "pitch_class_jaccard": _pitch_class_jaccard(source, target),
    }
    unavailable = [
        f"{relation}:{source_id}->{target_id}:{name}"
        for name, value in metrics.items()
        if value is None
    ]
    return (
        {
            "relation": relation,
            "source_id": source_id,
            "target_id": target_id,
            "variation_kind": variation_kind,
            **metrics,
        },
        unavailable,
    )


def _hold_metrics(composition: Composition) -> dict[str, Any]:
    expanded = _expand_use_range(composition, "whole", 0, len(composition.form))
    velocities = [note.velocity for note in expanded.notes]
    return {
        "duration_ms": expanded.duration_ms,
        "note_count": len(expanded.notes),
        "velocity_median": _round(statistics.median(velocities)) if velocities else None,
        "mean_polyphony": (
            _round(sum(note.duration_ms for note in expanded.notes) / expanded.duration_ms)
            if expanded.duration_ms > 0
            else None
        ),
        "pitch_classes": sorted({note.pitch % 12 for note in expanded.notes}),
    }


def describe_song_change(composition: Composition) -> dict[str, Any]:
    """階層で宣言した関係を、実音の複数記述値として返す。"""
    hold = _hold_metrics(composition)
    if not composition.parts or not composition.phrases:
        return {
            "schema_version": 1,
            "status": "unable_to_investigate",
            "hold": hold,
            "phrase_relations": [],
            "part_relations": [],
            "unavailable_metrics": ["parts_and_phrases"],
        }

    phrase_by_id = {phrase.phrase_id: phrase for phrase in composition.phrases}
    phrase_materials = {
        phrase.phrase_id: _expand_use_range(
            composition,
            phrase.phrase_id,
            phrase.start_use_index,
            phrase.end_use_index,
        )
        for phrase in composition.phrases
    }
    phrase_relations: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for index, phrase in enumerate(composition.phrases):
        if phrase.role in {"variation", "return"} and phrase.derived_from is not None:
            source = phrase_by_id[phrase.derived_from]
            report, missing = _relation(
                phrase_materials[source.phrase_id],
                phrase_materials[phrase.phrase_id],
                relation=phrase.role,
                source_id=source.phrase_id,
                target_id=phrase.phrase_id,
                variation_kind=phrase.variation_kind,
            )
            phrase_relations.append(report)
            unavailable.extend(missing)
        elif phrase.role == "contrast":
            prior_statement = next(
                (
                    candidate
                    for candidate in reversed(composition.phrases[:index])
                    if candidate.role == "statement"
                ),
                None,
            )
            if prior_statement is None:
                unavailable.append(f"contrast:{phrase.phrase_id}:source_statement")
                continue
            report, missing = _relation(
                phrase_materials[prior_statement.phrase_id],
                phrase_materials[phrase.phrase_id],
                relation="contrast",
                source_id=prior_statement.phrase_id,
                target_id=phrase.phrase_id,
            )
            phrase_relations.append(report)
            unavailable.extend(missing)

    part_materials = {
        part.part_id: _expand_use_range(
            composition,
            part.part_id,
            part.start_use_index,
            part.end_use_index,
        )
        for part in composition.parts
    }
    part_relations: list[dict[str, Any]] = []
    for left, right in combinations(composition.parts, 2):
        report, missing = _relation(
            part_materials[left.part_id],
            part_materials[right.part_id],
            relation="part_pair",
            source_id=left.part_id,
            target_id=right.part_id,
        )
        part_relations.append(report)
        unavailable.extend(missing)

    if hold["velocity_median"] is None:
        unavailable.append("hold:velocity_median")
    return {
        "schema_version": 1,
        "status": "measured_with_unavailable_metrics" if unavailable else "measured",
        "hold": hold,
        "phrase_relations": phrase_relations,
        "part_relations": part_relations,
        "unavailable_metrics": sorted(unavailable),
    }
