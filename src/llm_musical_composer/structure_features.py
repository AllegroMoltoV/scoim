"""拍を仮定せず、曲内の位置を保った構造特徴を抽出する。"""

from __future__ import annotations

import math
import statistics
from difflib import SequenceMatcher
from itertools import pairwise
from typing import Any

from llm_musical_composer.pilot_features import NoteEvent


def _round(value: float) -> float:
    return round(float(value), 8)


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _direction_change_rate(pitches: list[int]) -> float:
    intervals = [second - first for first, second in pairwise(pitches)]
    signs = [(interval > 0) - (interval < 0) for interval in intervals if interval]
    if len(signs) < 2:
        return 0.0
    return _round(sum(first != second for first, second in pairwise(signs)) / (len(signs) - 1))


def _polyphony(notes: list[NoteEvent], start: float, end: float) -> tuple[float, int]:
    changes: list[tuple[float, int]] = []
    occupied = 0.0
    for note in notes:
        note_end = note.onset_ms + note.duration_ms
        clipped_start = max(float(note.onset_ms), start)
        clipped_end = min(float(note_end), end)
        if clipped_end <= clipped_start:
            continue
        occupied += clipped_end - clipped_start
        changes.append((clipped_start, 1))
        changes.append((clipped_end, -1))
    current = 0
    maximum = 0
    for _, delta in sorted(changes, key=lambda item: (item[0], item[1])):
        current += delta
        maximum = max(maximum, current)
    width = end - start
    return (_round(occupied / width) if width > 0 else 0.0, maximum)


def _window_features(
    notes: list[NoteEvent],
    *,
    start: float,
    end: float,
    sounding_span: float,
    onset_notes: list[NoteEvent] | None = None,
) -> dict[str, float]:
    if onset_notes is None:
        onset_notes = [note for note in notes if start <= note.onset_ms < end]
    ordered = sorted(onset_notes, key=lambda note: (note.onset_ms, note.pitch, note.duration_ms))
    pitches = [note.pitch for note in ordered]
    onsets = sorted({note.onset_ms for note in ordered})
    iois = [float(second - first) / sounding_span for first, second in pairwise(onsets)]
    polyphony_mean, polyphony_max = _polyphony(notes, start, end)
    return {
        "onset_count": float(len(ordered)),
        "duration_median": _round(
            _median([float(note.duration_ms) / sounding_span for note in ordered])
        ),
        "velocity_median": _round(_median([float(note.velocity) for note in ordered])),
        "polyphony_mean": polyphony_mean,
        "polyphony_max": float(polyphony_max),
        "pitch_median": _round(_median([float(pitch) for pitch in pitches])),
        "pitch_range": float(max(pitches) - min(pitches)) if pitches else 0.0,
        "ioi_median": _round(_median(iois)),
        "pitch_direction_change_rate": _direction_change_rate(pitches),
    }


def _novelty(windows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = tuple(windows[0]["features"])
    series: dict[str, list[float]] = {}
    thresholds: dict[str, float] = {}
    for name in metric_names:
        values = [window["features"][name] for window in windows]
        differences = [_round(abs(second - first)) for first, second in pairwise(values)]
        series[name] = differences
        positive = [value for value in differences if value > 0]
        thresholds[name] = _round(_percentile(positive, 0.75)) if positive else 0.0
    candidates: list[dict[str, Any]] = []
    for boundary_index in range(1, len(windows)):
        supporting = [
            name
            for name in metric_names
            if thresholds[name] > 0 and series[name][boundary_index - 1] >= thresholds[name]
        ]
        if len(supporting) >= 3:
            candidates.append(
                {
                    "boundary_index": boundary_index,
                    "position": windows[boundary_index]["start"],
                    "supporting_metrics": supporting,
                }
            )
    return {"series": series, "thresholds": thresholds, "consensus_candidates": candidates}


def _sequence_similarity(first: tuple[Any, ...], second: tuple[Any, ...]) -> float:
    if not first or not second:
        return 0.0
    return _round(SequenceMatcher(a=first, b=second, autojunk=False).ratio())


def _pitch_signature(notes: list[NoteEvent]) -> tuple[int, ...]:
    pitches = [note.pitch for note in sorted(notes, key=lambda note: (note.onset_ms, note.pitch))]
    return tuple(second - first for first, second in pairwise(pitches))


def _timing_signature(
    notes: list[NoteEvent], window_width: float
) -> tuple[tuple[float, float], ...]:
    ordered = sorted(notes, key=lambda note: (note.onset_ms, note.pitch))
    if not ordered or window_width <= 0:
        return ()
    unique_onsets = sorted({note.onset_ms for note in ordered})
    gaps = {second: second - first for first, second in pairwise(unique_onsets)}
    first_onset = unique_onsets[0]
    return tuple(
        (
            _round(gaps.get(note.onset_ms, note.onset_ms - first_onset) / window_width),
            _round(note.duration_ms / window_width),
        )
        for note in ordered
    )


def _texture_similarity(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    if not any(first) or not any(second):
        return 0.0
    differences = [
        abs(left - right) / max(abs(left), abs(right), 1.0)
        for left, right in zip(first, second, strict=True)
    ]
    return _round(max(0.0, 1.0 - statistics.mean(differences)))


def _matrix(signatures: list[Any], similarity: Any) -> list[list[float]]:
    result: list[list[float]] = []
    for row, first in enumerate(signatures):
        values: list[float] = []
        for column, second in enumerate(signatures):
            values.append(1.0 if row == column else similarity(first, second))
        result.append(values)
    return result


def _repetition_summary(matrix: list[list[float]], *, threshold: float = 0.8) -> dict[str, Any]:
    contiguous: list[dict[str, Any]] = []
    delayed: list[dict[str, Any]] = []
    covered: set[int] = set()
    for row in range(len(matrix)):
        for column in range(row + 1, len(matrix)):
            similarity = matrix[row][column]
            if similarity < threshold:
                continue
            item = {
                "first_window": row,
                "second_window": column,
                "distance": column - row,
                "similarity": similarity,
            }
            if column == row + 1:
                contiguous.append(item)
            else:
                delayed.append(item)
                covered.update((row, column))
    return {
        "threshold": threshold,
        "contiguous_pairs": contiguous,
        "delayed_pairs": delayed,
        "delayed_coverage": _round(len(covered) / len(matrix)) if matrix else 0.0,
        "recurrence_distances": [item["distance"] for item in delayed],
    }


def _activity(windows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sources = {
        "onset_density": "onset_count",
        "velocity": "velocity_median",
        "polyphony": "polyphony_mean",
        "pitch_median": "pitch_median",
    }
    summaries: dict[str, Any] = {}
    high_indices: dict[str, set[int]] = {}
    for output_name, feature_name in sources.items():
        values = [float(window["features"][feature_name]) for window in windows]
        maximum = max(values)
        minimum = min(values)
        threshold = _percentile(values, 0.75)
        high_indices[output_name] = {
            index for index, value in enumerate(values) if value >= threshold and maximum > minimum
        }
        summaries[output_name] = {
            "values": [_round(value) for value in values],
            "minimum": _round(minimum),
            "maximum": _round(maximum),
            "range": _round(maximum - minimum),
            "peak_positions": [
                windows[index]["center"] for index, value in enumerate(values) if value == maximum
            ],
            "provisional_upper_quartile": _round(threshold),
        }
    candidates: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        supporting = [name for name, indices in high_indices.items() if index in indices]
        if len(supporting) >= 3:
            candidates.append(
                {
                    "window": index,
                    "position": window["center"],
                    "supporting_series": supporting,
                }
            )
    return summaries, candidates


def _declared_form_comparison(
    material_ids: list[str], matrices: dict[str, list[list[float]]]
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for first in range(len(material_ids)):
        for second in range(first + 1, len(material_ids)):
            if material_ids[first] != material_ids[second]:
                continue
            comparisons.append(
                {
                    "first_window": first,
                    "second_window": second,
                    "material_id": material_ids[first],
                    "same_declared_material": True,
                    "pitch_interval_similarity": matrices["pitch_interval"][first][second],
                    "relative_timing_similarity": matrices["relative_timing"][first][second],
                    "texture_similarity": matrices["texture"][first][second],
                }
            )
    return comparisons


def extract_structure_features(
    notes: list[NoteEvent],
    *,
    window_count: int = 16,
    declared_material_ids: list[str] | None = None,
    coordinate: str = "elapsed_time",
    phase_fraction: float = 0.0,
) -> dict[str, Any]:
    """構造の位置、反復、活動量を合算せず返す。"""
    if window_count < 2:
        raise ValueError("window_count must be at least two")
    if coordinate not in {"elapsed_time", "onset_order"}:
        raise ValueError("coordinate must be elapsed_time or onset_order")
    if not 0.0 <= phase_fraction < 1.0:
        raise ValueError("phase_fraction must be at least zero and less than one")
    if coordinate == "onset_order" and phase_fraction:
        raise ValueError("phase_fraction is only supported for elapsed_time")
    if declared_material_ids is not None and len(declared_material_ids) != window_count:
        raise ValueError("declared_material_ids must match window_count")
    if declared_material_ids is not None and phase_fraction:
        raise ValueError("declared_material_ids cannot be used with a shifted phase")
    usable = [note for note in notes if note.duration_ms > 0]
    if not usable:
        return {
            "status": "unable_to_investigate",
            "detail": "no complete pitched notes",
            "windows": [],
        }
    sounding_start = min(note.onset_ms for note in usable)
    sounding_end = max(note.onset_ms + note.duration_ms for note in usable)
    sounding_span = float(sounding_end - sounding_start)
    if sounding_span <= 0:
        return {
            "status": "unable_to_investigate",
            "detail": "sounding duration is zero",
            "windows": [],
        }
    windows: list[dict[str, Any]] = []
    notes_by_window: list[list[NoteEvent]] = []
    window_widths: list[float] = []
    if coordinate == "elapsed_time":
        slice_count = window_count if phase_fraction == 0 else window_count - 1
        slices = []
        for index in range(slice_count):
            normalized_start = (index + phase_fraction) / window_count
            normalized_end = (index + phase_fraction + 1) / window_count
            start = sounding_start + sounding_span * normalized_start
            feature_end = sounding_start + sounding_span * normalized_end
            selection_end = (
                sounding_end + 1
                if phase_fraction == 0 and index == window_count - 1
                else feature_end
            )
            slices.append(
                (
                    index,
                    normalized_start,
                    normalized_end,
                    start,
                    feature_end,
                    [note for note in usable if start <= note.onset_ms < selection_end],
                )
            )
    else:
        unique_onsets = sorted({note.onset_ms for note in usable})
        onset_rank = {onset: index for index, onset in enumerate(unique_onsets)}
        slices = []
        for index in range(window_count):
            first_rank = math.floor(index * len(unique_onsets) / window_count)
            next_rank = math.floor((index + 1) * len(unique_onsets) / window_count)
            selected = [
                note for note in usable if first_rank <= onset_rank[note.onset_ms] < next_rank
            ]
            if first_rank < len(unique_onsets):
                start = float(unique_onsets[first_rank])
            else:
                start = float(sounding_end)
            if next_rank < len(unique_onsets):
                feature_end = float(unique_onsets[next_rank])
            else:
                feature_end = float(sounding_end)
            if feature_end <= start:
                feature_end = start + 1.0
            slices.append(
                (
                    index,
                    index / window_count,
                    (index + 1) / window_count,
                    start,
                    feature_end,
                    selected,
                )
            )

    for index, normalized_start, normalized_end, start, feature_end, onset_notes in slices:
        notes_by_window.append(onset_notes)
        window_widths.append(feature_end - start)
        windows.append(
            {
                "index": index,
                "start": _round(normalized_start),
                "end": _round(normalized_end),
                "center": _round((normalized_start + normalized_end) / 2),
                "features": _window_features(
                    usable,
                    start=start,
                    end=feature_end,
                    sounding_span=sounding_span,
                    onset_notes=onset_notes,
                ),
            }
        )
    pitch_signatures = [_pitch_signature(window_notes) for window_notes in notes_by_window]
    timing_signatures = [
        _timing_signature(window_notes, window_width)
        for window_notes, window_width in zip(notes_by_window, window_widths, strict=True)
    ]
    texture_signatures = [
        (
            window["features"]["onset_count"],
            window["features"]["polyphony_mean"],
            window["features"]["polyphony_max"],
            window["features"]["pitch_range"],
        )
        for window in windows
    ]
    matrices = {
        "pitch_interval": _matrix(pitch_signatures, _sequence_similarity),
        "relative_timing": _matrix(timing_signatures, _sequence_similarity),
        "texture": _matrix(texture_signatures, _texture_similarity),
    }
    activity, high_activity = _activity(windows)
    result = {
        "status": "pass",
        "window_count": window_count,
        "coordinate": coordinate,
        "phase_fraction": phase_fraction,
        "windows": windows,
        "novelty": _novelty(windows),
        "similarity_matrices": matrices,
        "repetition": {name: _repetition_summary(matrix) for name, matrix in matrices.items()},
        "activity": activity,
        "high_activity_candidates": high_activity,
    }
    if declared_material_ids is not None:
        result["declared_form_comparison"] = _declared_form_comparison(
            declared_material_ids, matrices
        )
    return result


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "available_count": len(values),
        "p25": _round(_percentile(values, 0.25)),
        "median": _round(_percentile(values, 0.5)),
        "p75": _round(_percentile(values, 0.75)),
    }


def build_structure_reference_profile(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """曲ごとの配置を平均せず、構造要約の分布だけを作る。"""
    usable = [report for report in reports if report.get("status") == "pass"]
    if not usable:
        raise ValueError("at least one usable structure report is required")
    boundary_counts = [float(len(report["novelty"]["consensus_candidates"])) for report in usable]
    repetition: dict[str, Any] = {}
    for modality in ("pitch_interval", "relative_timing", "texture"):
        coverages = [float(report["repetition"][modality]["delayed_coverage"]) for report in usable]
        recurrence = [
            float(statistics.median(report["repetition"][modality]["recurrence_distances"]))
            for report in usable
            if report["repetition"][modality]["recurrence_distances"]
        ]
        repetition[modality] = {
            "delayed_coverage": _summary(coverages),
            "median_recurrence_distance": _summary(recurrence)
            if recurrence
            else {"available_count": 0, "status": "unable_to_investigate"},
        }
    high_counts = [float(len(report["high_activity_candidates"])) for report in usable]
    first_positions = [
        float(report["high_activity_candidates"][0]["position"])
        for report in usable
        if report["high_activity_candidates"]
    ]
    return {
        "schema_version": 1,
        "source_count": len(usable),
        "axes": {
            "boundaries": {"consensus_candidate_count": _summary(boundary_counts)},
            "repetition": repetition,
            "activity": {
                "high_activity_candidate_count": _summary(high_counts),
                "first_high_activity_position": _summary(first_positions)
                if first_positions
                else {"available_count": 0, "status": "unable_to_investigate"},
            },
        },
    }


def _control_notes(
    start: int, pitches: list[int], *, velocity: int, spacing: int
) -> list[NoteEvent]:
    return [
        NoteEvent(pitch, start + index * spacing, 300, velocity)
        for index, pitch in enumerate(pitches)
    ]


def run_structure_controls() -> dict[str, dict[str, str]]:
    """仮の構造指標が既知の変形へ反応するかを確認する。"""
    a = _control_notes(0, [60, 64, 62, 67], velocity=45, spacing=200)
    b = _control_notes(1_000, [48, 55, 60, 64, 67, 72], velocity=100, spacing=100)
    transposed_a = _control_notes(2_000, [67, 71, 69, 74], velocity=45, spacing=200)
    c = _control_notes(2_000, [60, 66, 61, 70], velocity=45, spacing=200)
    original_notes = a + b + transposed_a
    original = extract_structure_features(original_notes, window_count=3)
    stretched = extract_structure_features(
        [
            NoteEvent(note.pitch, note.onset_ms * 2, note.duration_ms * 2, note.velocity)
            for note in original_notes
        ],
        window_count=3,
    )
    reordered = extract_structure_features(
        _control_notes(0, [48, 55, 60, 64, 67, 72], velocity=100, spacing=100)
        + _control_notes(1_000, [60, 64, 62, 67], velocity=45, spacing=200)
        + transposed_a,
        window_count=3,
    )
    broken = extract_structure_features(a + b + c, window_count=3)
    flat = extract_structure_features(
        [NoteEvent(note.pitch, note.onset_ms, note.duration_ms, 65) for note in original_notes],
        window_count=3,
    )
    non_velocity_equal = all(
        flat["activity"][name] == original["activity"][name]
        for name in ("onset_density", "polyphony", "pitch_median")
    )
    outcomes = {
        "time_stretch": original == stretched,
        "section_reorder": (
            original["activity"]["velocity"]["peak_positions"]
            != reordered["activity"]["velocity"]["peak_positions"]
            or original["novelty"] != reordered["novelty"]
        ),
        "repetition_break": (
            original["similarity_matrices"]["pitch_interval"][0][2]
            > broken["similarity_matrices"]["pitch_interval"][0][2]
        ),
        "velocity_flatten": (
            flat["activity"]["velocity"]["range"] < original["activity"]["velocity"]["range"]
            and non_velocity_equal
        ),
    }
    return {
        name: {
            "status": "pass" if passed else "fail",
            "detail": "counterexample behaved as expected"
            if passed
            else "counterexample did not produce the expected change",
        }
        for name, passed in outcomes.items()
    }
