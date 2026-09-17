"""「はげしさ」の異なる実現方法を、合算せずに記述する。"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

from llm_musical_composer.composition_ir import Material, Note

LOW_BASS_MAX_PITCH = 48


def _round(value: float) -> float:
    return round(float(value), 8)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _voice_groups(notes: Sequence[Note]) -> dict[str, dict[int, list[Note]]]:
    groups: dict[str, dict[int, list[Note]]] = defaultdict(lambda: defaultdict(list))
    for note in notes:
        groups[note.voice or "unassigned"][note.at_ms].append(note)
    return {voice: dict(attacks) for voice, attacks in groups.items()}


def _voice_iois(groups: dict[str, dict[int, list[Note]]]) -> tuple[list[int], list[float]]:
    iois: list[int] = []
    consecutive_changes: list[float] = []
    for attacks in groups.values():
        onsets = sorted(attacks)
        current_iois = [right - left for left, right in pairwise(onsets)]
        iois.extend(current_iois)
        consecutive_changes.extend(
            abs(math.log2(right / left))
            for left, right in pairwise(current_iois)
            if left > 0 and right > 0
        )
    return iois, consecutive_changes


def _articulation_ratios(groups: dict[str, dict[int, list[Note]]]) -> list[float]:
    ratios: list[float] = []
    for attacks in groups.values():
        onsets = sorted(attacks)
        for left, right in pairwise(onsets):
            representative_duration = statistics.median(note.duration_ms for note in attacks[left])
            ratios.append(representative_duration / (right - left))
    return ratios


def _pitch_repetition(groups: dict[str, dict[int, list[Note]]]) -> tuple[int, int]:
    repeated = 0
    comparisons = 0
    for attacks in groups.values():
        pitch_sets = [
            tuple(sorted(note.pitch for note in attacks[onset])) for onset in sorted(attacks)
        ]
        for left, right in pairwise(pitch_sets):
            comparisons += 1
            repeated += left == right
    return repeated, comparisons


def _merge_interval_duration(intervals: Sequence[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _validate(material: Material) -> None:
    if material.duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    for note in material.notes:
        if note.at_ms < 0:
            raise ValueError(f"note {note.event_id} at_ms must not be negative")
        if note.duration_ms <= 0:
            raise ValueError(f"note {note.event_id} duration_ms must be positive")


def extract_intensity_realization(material: Material) -> dict[str, Any]:
    """素材の時間、強弱、音価、反復、声部関係、低音を別々に返す。

    この関数は総合点、目標値、合否を返さない。利用者が感じる「はげしさ」と各記述値の
    対応は、比較音源による検証前には確定できないためである。
    """
    _validate(material)
    notes = tuple(material.notes)
    groups = _voice_groups(notes)
    unavailable: list[str] = []

    iois, ioi_changes = _voice_iois(groups)
    median_ioi = _round(statistics.median(iois)) if iois else None
    if median_ioi is None:
        unavailable.append("median_voice_local_ioi_ms")
    median_ioi_change = _round(statistics.median(ioi_changes)) if ioi_changes else None
    if median_ioi_change is None:
        unavailable.append("median_consecutive_ioi_change_log2")

    velocities = [note.velocity for note in notes]
    velocity_median = _round(statistics.median(velocities)) if velocities else None
    accent_span = (
        _round(_percentile(velocities, 0.9) - statistics.median(velocities)) if velocities else None
    )
    if accent_span is None:
        unavailable.extend(("velocity_median", "velocity_p90_minus_median"))

    articulation = _articulation_ratios(groups)
    articulation_median = _round(statistics.median(articulation)) if articulation else None
    if articulation_median is None:
        unavailable.append("median_duration_to_next_onset_ratio")

    repeated, repetition_comparisons = _pitch_repetition(groups)
    repetition_ratio = _round(repeated / repetition_comparisons) if repetition_comparisons else None
    if repetition_ratio is None:
        unavailable.append("consecutive_pitch_set_repeat_ratio")

    upper_onsets = set(groups.get("upper", {}))
    lower_onsets = set(groups.get("lower", {}))
    if upper_onsets and lower_onsets:
        shared_onsets = upper_onsets & lower_onsets
        all_onsets = upper_onsets | lower_onsets
        synchronous_ratio = _round(len(shared_onsets) / len(all_onsets))
        coupled = sum(
            bool(
                {note.pitch % 12 for note in groups["upper"][onset]}
                & {note.pitch % 12 for note in groups["lower"][onset]}
            )
            for onset in shared_onsets
        )
        if shared_onsets:
            pitch_class_coupling = _round(coupled / len(shared_onsets))
        else:
            pitch_class_coupling = None
            unavailable.append("pitch_class_coupling_ratio")
    else:
        synchronous_ratio = None
        pitch_class_coupling = None
        unavailable.extend(("synchronous_onset_ratio", "pitch_class_coupling_ratio"))

    low_bass = [
        note for note in notes if note.voice == "lower" and note.pitch <= LOW_BASS_MAX_PITCH
    ]
    if low_bass:
        low_bass_intervals = [
            (max(0, note.at_ms), min(material.duration_ms, note.at_ms + note.duration_ms))
            for note in low_bass
            if note.at_ms < material.duration_ms
        ]
        sounding_duration = _merge_interval_duration(
            [(start, end) for start, end in low_bass_intervals if end > start]
        )
        bass_ratio = _round(sounding_duration / material.duration_ms)
        bass_velocity = _round(statistics.median(note.velocity for note in low_bass))
    else:
        bass_ratio = None
        bass_velocity = None
        unavailable.extend(("low_bass_sounding_time_ratio", "low_bass_velocity_median"))

    return {
        "schema_version": 1,
        "status": "measured_with_unavailable_metrics" if unavailable else "measured",
        "hold": {
            "duration_ms": material.duration_ms,
            "note_count": len(notes),
            "global_onset_count": len({note.at_ms for note in notes}),
        },
        "timing": {
            "median_voice_local_ioi_ms": median_ioi,
            "median_consecutive_ioi_change_log2": median_ioi_change,
        },
        "accent": {
            "velocity_median": velocity_median,
            "velocity_p90_minus_median": accent_span,
        },
        "articulation": {
            "median_duration_to_next_onset_ratio": articulation_median,
        },
        "repetition": {
            "consecutive_pitch_set_repeat_ratio": repetition_ratio,
        },
        "voice_coupling": {
            "synchronous_onset_ratio": synchronous_ratio,
            "pitch_class_coupling_ratio": pitch_class_coupling,
        },
        "bass": {
            "low_bass_sounding_time_ratio": bass_ratio,
            "low_bass_velocity_median": bass_velocity,
        },
        "unavailable_metrics": sorted(unavailable),
    }
