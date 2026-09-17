"""上声・下声の独立性と、型付き変奏の契約を実イベントから評価する。"""

from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher
from itertools import pairwise
from statistics import median
from typing import Any

from llm_musical_composer.composition_ir import Composition, Material, Note

VOICE_LABELS = ("upper", "lower")
ONSET_TOLERANCE_MS = 50


def _within_active(note: Note, other: Note) -> bool:
    return other.at_ms <= note.at_ms < other.at_ms + other.duration_ms


def _near_fraction(first: list[int], second: list[int], tolerance_ms: int) -> float:
    if not first or not second:
        return 0.0
    matched = sum(any(abs(value - other) <= tolerance_ms for other in second) for value in first)
    return matched / len(first)


def _active_fraction(first: list[Note], second: list[Note]) -> float:
    if not first or not second:
        return 0.0
    return sum(any(_within_active(note, other) for other in second) for note in first) / len(first)


def voice_texture_features(
    notes: Iterable[Note], *, tolerance_ms: int = ONSET_TOLERANCE_MS
) -> dict[str, Any]:
    """音符列から、二声が同時に存在しながら別々に動く度合いを返す。"""
    note_list = tuple(notes)
    issues: list[str] = []
    invalid = [note.event_id for note in note_list if note.voice not in VOICE_LABELS]
    if invalid:
        issues.append("all notes must label voice as upper or lower")
    by_voice = {
        voice: [note for note in note_list if note.voice == voice] for voice in VOICE_LABELS
    }
    if any(not by_voice[voice] for voice in VOICE_LABELS):
        issues.append("both voices must contain notes")

    upper = by_voice["upper"]
    lower = by_voice["lower"]
    upper_onsets = sorted({note.at_ms for note in upper})
    lower_onsets = sorted({note.at_ms for note in lower})
    coupling = (
        _near_fraction(upper_onsets, lower_onsets, tolerance_ms)
        + _near_fraction(lower_onsets, upper_onsets, tolerance_ms)
    ) / 2
    cross_active = (_active_fraction(upper, lower) + _active_fraction(lower, upper)) / 2
    pitch_separation = (
        float(median(note.pitch for note in upper) - median(note.pitch for note in lower))
        if upper and lower
        else 0.0
    )

    if upper and lower:
        if len(upper_onsets) < 2 or len(lower_onsets) < 2:
            issues.append("both voices must have at least two distinct onsets")
        if coupling < 0.10:
            issues.append("voices never meet at phrase anchors")
        if coupling > 0.85:
            issues.append("voices are too tightly coupled in lockstep")
        if cross_active < 0.25:
            issues.append("voices are temporally disjoint")
        if pitch_separation < 5:
            issues.append("upper and lower registers are not separated")

    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "note_counts": {voice: len(by_voice[voice]) for voice in VOICE_LABELS},
        "unique_onset_counts": {
            "upper": len(upper_onsets),
            "lower": len(lower_onsets),
        },
        "near_onset_coupling": coupling,
        "cross_voice_active_ratio": cross_active,
        "pitch_median_separation": pitch_separation,
        "duration_medians_ms": {
            voice: (
                float(median(note.duration_ms for note in by_voice[voice]))
                if by_voice[voice]
                else None
            )
            for voice in VOICE_LABELS
        },
    }


def _expanded_notes(composition: Composition) -> tuple[Note, ...]:
    offset = 0
    expanded: list[Note] = []
    for use in composition.form:
        material = composition.material_by_id[use.material_id]
        expanded.extend(
            Note(
                note.event_id,
                note.at_ms + offset,
                note.duration_ms,
                note.pitch,
                note.velocity,
                note.voice,
            )
            for note in material.notes
        )
        offset += material.duration_ms
    return tuple(expanded)


def evaluate_piano_texture(composition: Composition) -> dict[str, Any]:
    """全曲と各フレーズについて、可聴上の二声性を検査する。"""
    issues: list[str] = []
    phrase_reports: list[dict[str, Any]] = []
    for phrase in composition.phrases:
        notes: list[Note] = []
        for use in composition.form[phrase.start_use_index : phrase.end_use_index]:
            notes.extend(composition.material_by_id[use.material_id].notes)
        voices = {note.voice for note in notes}
        status = "pass" if set(VOICE_LABELS) <= voices else "fail"
        if status == "fail":
            issues.append(f"phrase {phrase.phrase_id} must contain both voices")
        phrase_reports.append(
            {
                "phrase_id": phrase.phrase_id,
                "status": status,
                "voice_counts": {
                    voice: sum(note.voice == voice for note in notes) for voice in VOICE_LABELS
                },
            }
        )

    overall = voice_texture_features(_expanded_notes(composition))
    issues.extend(overall["issues"])
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "overall": overall,
        "phrases": phrase_reports,
    }


def _voice_notes(material: Material, voice: str) -> list[Note]:
    return sorted(
        (note for note in material.notes if note.voice == voice),
        key=lambda note: (note.at_ms, note.pitch, note.duration_ms),
    )


def _top_line(material: Material, voice: str) -> list[Note]:
    grouped: dict[int, list[Note]] = {}
    for note in _voice_notes(material, voice):
        grouped.setdefault(note.at_ms, []).append(note)
    return [max(grouped[onset], key=lambda note: note.pitch) for onset in sorted(grouped)]


def _intervals(notes: list[Note]) -> tuple[int, ...]:
    return tuple(second.pitch - first.pitch for first, second in pairwise(notes))


def _directions(notes: list[Note]) -> tuple[int, ...]:
    return tuple(
        1 if second.pitch > first.pitch else -1 if second.pitch < first.pitch else 0
        for first, second in pairwise(notes)
    )


def _rhythm(notes: list[Note]) -> tuple[int, ...]:
    if not notes:
        return ()
    first = notes[0].at_ms
    span = notes[-1].at_ms - first
    if span <= 0:
        return (0,) * len(notes)
    return tuple(round((note.at_ms - first) * 16 / span) for note in notes)


def _similarity(first: tuple[int, ...], second: tuple[int, ...]) -> float:
    if not first or not second:
        return 0.0
    return SequenceMatcher(a=first, b=second, autojunk=False).ratio()


def _variation_pair(source: Material, variation: Material, kind: str) -> dict[str, Any]:
    issues: list[str] = []
    missing_voice = any(
        not _voice_notes(material, voice)
        for material in (source, variation)
        for voice in VOICE_LABELS
    )
    if missing_voice:
        issues.append("source and variation must both contain both voices")

    source_upper = _top_line(source, "upper")
    variation_upper = _top_line(variation, "upper")
    source_lower = _top_line(source, "lower")
    variation_lower = _top_line(variation, "lower")
    upper_contour = _similarity(_intervals(source_upper), _intervals(variation_upper))
    upper_direction = _similarity(_directions(source_upper), _directions(variation_upper))
    upper_rhythm = _similarity(_rhythm(source_upper), _rhythm(variation_upper))
    lower_rhythm = _similarity(_rhythm(source_lower), _rhythm(variation_lower))
    lower_contour = _similarity(_intervals(source_lower), _intervals(variation_lower))
    register_shift = (
        abs(
            float(
                median(note.pitch for note in variation_upper)
                - median(note.pitch for note in source_upper)
            )
        )
        if source_upper and variation_upper
        else 0.0
    )

    if kind == "rhythmic":
        if upper_contour < 0.50:
            issues.append("rhythmic variation must preserve the upper contour")
        if upper_rhythm > 0.80 and lower_rhythm > 0.80:
            issues.append("rhythmic variation must audibly change rhythm")
    elif kind == "textural":
        if upper_contour < 0.50 or upper_rhythm < 0.75:
            issues.append("textural variation must preserve the upper motif")
        if lower_rhythm > 0.80 and lower_contour > 0.80:
            issues.append("textural variation must change the lower layer")
    elif kind == "registral":
        if upper_direction < 0.80 or upper_rhythm < 0.80:
            issues.append("registral variation must preserve rhythm and contour")
        if register_shift < 5:
            issues.append("registral variation must move the upper register")
    else:
        issues.append("variation_kind must be rhythmic, textural, or registral")

    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "source_material_id": source.material_id,
        "variation_material_id": variation.material_id,
        "variation_kind": kind,
        "upper_contour_similarity": upper_contour,
        "upper_direction_similarity": upper_direction,
        "upper_rhythm_similarity": upper_rhythm,
        "lower_rhythm_similarity": lower_rhythm,
        "lower_contour_similarity": lower_contour,
        "upper_register_shift": register_shift,
    }


def evaluate_variation_contracts(composition: Composition) -> dict[str, Any]:
    """宣言された変奏型ごとに、保持要素と変更要素を別々に検査する。"""
    phrase_by_id = {phrase.phrase_id: phrase for phrase in composition.phrases}
    material_map = composition.material_by_id
    reports: list[dict[str, Any]] = []
    issues: list[str] = []
    for phrase in composition.phrases:
        if phrase.role != "variation":
            continue
        if phrase.variation_kind is None:
            issues.append(f"variation phrase {phrase.phrase_id} requires variation_kind")
            continue
        assert phrase.derived_from is not None
        source_phrase = phrase_by_id[phrase.derived_from]
        source_ids = {
            use.material_id
            for use in composition.form[source_phrase.start_use_index : source_phrase.end_use_index]
            if use.role != "transition"
        }
        for use in composition.form[phrase.start_use_index : phrase.end_use_index]:
            if use.role == "transition":
                continue
            variation = material_map[use.material_id]
            if variation.derived_from not in source_ids:
                continue
            report = _variation_pair(
                material_map[variation.derived_from], variation, phrase.variation_kind
            )
            reports.append(report)
            issues.extend(f"{variation.material_id}: {issue}" for issue in report["issues"])
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "pairs": reports,
    }
