"""最小縦断試験で使う三領域の暫定特徴量。"""

from __future__ import annotations

import enum
import math
import statistics
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from llm_musical_composer.smf_notes import load_smf_notes


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    onset_ms: int
    duration_ms: int
    velocity: int


class FeatureStatus(enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    UNABLE = "unable_to_investigate"


def _round(value: float) -> float:
    return round(float(value), 8)


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _changes(values: list[float]) -> float:
    return _median([abs(second - first) for first, second in pairwise(values)])


def _direction_change(values: list[int]) -> float:
    signs = [(value > 0) - (value < 0) for value in values if value]
    if len(signs) < 2:
        return 0.0
    return sum(first != second for first, second in pairwise(signs)) / (len(signs) - 1)


def _maximum_polyphony(notes: list[NoteEvent]) -> int:
    changes: list[tuple[int, int]] = []
    for note in notes:
        changes.append((note.onset_ms, 1))
        changes.append((note.onset_ms + note.duration_ms, -1))
    current = 0
    maximum = 0
    for _, delta in sorted(changes, key=lambda item: (item[0], item[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


def extract_features(notes: list[NoteEvent]) -> dict[str, dict[str, float]]:
    """拍を仮定せず、音列と曲内相対時間から特徴量を作る。"""
    ordered = sorted(notes, key=lambda note: (note.onset_ms, note.pitch, note.duration_ms))
    pitches = [note.pitch for note in ordered]
    intervals = [second - first for first, second in pairwise(pitches)]
    durations = [float(note.duration_ms) for note in ordered]
    onsets = sorted({note.onset_ms for note in ordered})
    iois = [float(second - first) for first, second in pairwise(onsets)]
    duration_median = _median([value for value in durations if value > 0]) or 1.0
    ioi_median = _median([value for value in iois if value > 0]) or 1.0
    duration_ratios = [value / duration_median for value in durations]
    ioi_ratios = [value / ioi_median for value in iois]
    span = max(
        (note.onset_ms + note.duration_ms for note in ordered),
        default=0,
    ) - min((note.onset_ms for note in ordered), default=0)
    normalized_span = span / duration_median if duration_median else 0

    return {
        "pitch_order": {
            "median_abs_interval": _round(_median([abs(value) for value in intervals])),
            "interval_change": _round(_changes([float(value) for value in intervals])),
            "direction_change_rate": _round(_direction_change(intervals)),
        },
        "relative_timing": {
            "duration_change": _round(_changes(duration_ratios)),
            "ioi_change": _round(_changes(ioi_ratios)),
            "median_duration_to_ioi": _round(duration_median / ioi_median),
        },
        "texture": {
            "pitch_range": _round(max(pitches) - min(pitches) if pitches else 0),
            "maximum_polyphony": _round(_maximum_polyphony(ordered)),
            "notes_per_normalized_span": _round(len(ordered) / normalized_span)
            if normalized_span
            else 0.0,
        },
    }


def extract_smf_features(path: Path) -> dict[str, dict[str, float]]:
    smf_notes = load_smf_notes(path)
    notes = [
        NoteEvent(note.pitch, note.onset_ms, note.duration_ms, note.velocity) for note in smf_notes
    ]
    if len(notes) < 2:
        raise ValueError("at least two complete pitched notes are required")
    return extract_features(notes)


def build_reference_profile_from_directory(
    input_dir: Path, *, excluded_names: frozenset[str] = frozenset()
) -> dict[str, Any]:
    feature_sets: list[dict[str, dict[str, float]]] = []
    source_files: list[str] = []
    unable: list[dict[str, str]] = []
    excluded_casefold = {name.casefold() for name in excluded_names}
    for path in sorted(Path(input_dir).glob("*.mid"), key=lambda item: item.name.casefold()):
        if path.name.casefold() in excluded_casefold:
            continue
        try:
            feature_sets.append(extract_smf_features(path))
            source_files.append(path.name)
        except (OSError, ValueError, EOFError) as error:
            unable.append({"name": path.name, "error": f"{type(error).__name__}: {error}"})
    profile = build_reference_profile(feature_sets, source_count=len(source_files))
    profile["source_files"] = source_files
    profile["excluded_files"] = sorted(excluded_names)
    profile["unable_to_investigate"] = unable
    return profile


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_reference_profile(
    feature_sets: list[dict[str, dict[str, float]]], *, source_count: int
) -> dict[str, Any]:
    if not feature_sets:
        raise ValueError("at least one feature set is required")
    axes: dict[str, Any] = {}
    for axis in feature_sets[0]:
        axes[axis] = {}
        for metric in feature_sets[0][axis]:
            values = [features[axis][metric] for features in feature_sets]
            axes[axis][metric] = {
                "p25": _round(_percentile(values, 0.25)),
                "median": _round(_percentile(values, 0.5)),
                "p75": _round(_percentile(values, 0.75)),
            }
    return {"schema_version": 1, "source_count": source_count, "axes": axes}


def evaluate_features(
    features: dict[str, dict[str, float | None]], profile: dict[str, Any]
) -> dict[str, Any]:
    axes: dict[str, Any] = {}
    for axis in profile["axes"]:
        metrics = features[axis]
        metric_results: dict[str, Any] = {}
        for metric, bounds in profile["axes"][axis].items():
            value = metrics.get(metric)
            if value is None or not all(key in bounds for key in ("p25", "median", "p75")):
                metric_results[metric] = {
                    "value": value,
                    "inside_iqr": False,
                    "normalized_distance": None,
                    "status": FeatureStatus.UNABLE.value,
                }
                continue
            iqr = bounds["p75"] - bounds["p25"]
            scale = iqr if iqr > 1e-9 else max(abs(bounds["median"]), 1.0)
            if value < bounds["p25"]:
                distance = (bounds["p25"] - value) / scale
            elif value > bounds["p75"]:
                distance = (value - bounds["p75"]) / scale
            else:
                distance = 0.0
            metric_results[metric] = {
                "value": value,
                "inside_iqr": distance == 0,
                "normalized_distance": _round(distance),
                "status": FeatureStatus.PASS.value,
            }
        available_distances = [
            item["normalized_distance"]
            for item in metric_results.values()
            if item["normalized_distance"] is not None
        ]
        axes[axis] = {
            "inside_iqr": all(item["inside_iqr"] for item in metric_results.values()),
            "normalized_distance": (
                _round(statistics.mean(available_distances)) if available_distances else 0.0
            ),
            "metrics": metric_results,
        }
    return {
        "axes": axes,
        "inside_axis_count": sum(item["inside_iqr"] for item in axes.values()),
        "worst_axis_distance": max(
            (item["normalized_distance"] for item in axes.values()), default=0.0
        ),
    }


def run_minimal_controls(notes: list[NoteEvent]) -> dict[str, Any]:
    original = extract_features(notes)
    stretched = extract_features(
        [
            NoteEvent(
                pitch=note.pitch,
                onset_ms=note.onset_ms * 2,
                duration_ms=note.duration_ms * 2,
                velocity=note.velocity,
            )
            for note in notes
        ]
    )
    timing_equal = original["relative_timing"] == stretched["relative_timing"]
    if len(notes) < 4:
        order_status = FeatureStatus.UNABLE
        order_detail = "fewer than four notes"
    else:
        ordered = sorted(notes, key=lambda note: (note.onset_ms, note.pitch))
        pitches = sorted(note.pitch for note in ordered)
        alternating = pitches[::2] + list(reversed(pitches[1::2]))
        durations = list(reversed([note.duration_ms for note in ordered]))
        disrupted = [
            NoteEvent(alternating[index], note.onset_ms, durations[index], note.velocity)
            for index, note in enumerate(ordered)
        ]
        altered = extract_features(disrupted)
        changed_axes = [
            axis for axis in ("pitch_order", "relative_timing") if altered[axis] != original[axis]
        ]
        order_changed = bool(changed_axes)
        order_status = FeatureStatus.PASS if order_changed else FeatureStatus.FAIL
        order_detail = "order-sensitive features changed" if order_changed else "no change detected"
    reuse_passed = len(set(("A", "B", "A"))) < 3 and len(set(("A", "B", "C"))) == 3
    return {
        "time_stretch": {
            "status": (FeatureStatus.PASS if timing_equal else FeatureStatus.FAIL).value,
            "detail": "relative timing preserved" if timing_equal else "relative timing changed",
        },
        "order_disruption": {
            "status": order_status.value,
            "detail": order_detail,
            "changed_axes": changed_axes if len(notes) >= 4 else [],
        },
        "material_reuse": {
            "status": (FeatureStatus.PASS if reuse_passed else FeatureStatus.FAIL).value,
            "detail": "A-B-A passes and A-B-C fails",
        },
    }
