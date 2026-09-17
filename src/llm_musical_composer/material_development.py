"""拍を仮定せず、短い素材内の音価、発音間隔、発音群の変化を測る。"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from llm_musical_composer.composition_ir import Material
from llm_musical_composer.smf_notes import SmfNote, load_smf_notes

METRICS = (
    "duration_change_rate",
    "ioi_change_rate",
    "attack_size_change_rate",
)
REFERENCE_WINDOW_LENGTHS_MS = (8_000, 11_000, 14_000)
ATTACK_TOLERANCE_SPAN_RATIO = 0.004


@dataclass(frozen=True)
class DevelopmentEvent:
    onset_ms: int
    duration_ms: int


def _round(value: float) -> float:
    return round(float(value), 8)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _change_rate(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    return sum(left != right for left, right in pairwise(values)) / (len(values) - 1)


def _ratio_bins(values: list[float]) -> list[int]:
    positive = [value for value in values if value > 0]
    if not positive:
        return []
    median = statistics.median(positive)
    return [round(math.log2(value / median) * 4) for value in positive]


def _attack_groups(events: list[DevelopmentEvent], *, span_ms: int) -> list[list[DevelopmentEvent]]:
    tolerance_ms = max(1, round(span_ms * ATTACK_TOLERANCE_SPAN_RATIO))
    groups: list[list[DevelopmentEvent]] = []
    group_start = 0
    for event in sorted(events, key=lambda item: (item.onset_ms, item.duration_ms)):
        if not groups or event.onset_ms - group_start > tolerance_ms:
            groups.append([event])
            group_start = event.onset_ms
        else:
            groups[-1].append(event)
    return groups


def extract_material_development(
    events: list[DevelopmentEvent], *, span_ms: int
) -> dict[str, float]:
    """素材内の変化率を時間伸縮に不変な三つの値として返す。"""
    if span_ms <= 0:
        raise ValueError("span_ms must be positive")
    groups = _attack_groups(events, span_ms=span_ms)
    durations = [statistics.median(event.duration_ms for event in group) for group in groups]
    group_onsets = [min(event.onset_ms for event in group) for group in groups]
    iois = [right - left for left, right in pairwise(group_onsets)]
    group_sizes = [len(group) for group in groups]
    return {
        "duration_change_rate": _round(_change_rate(_ratio_bins(durations))),
        "ioi_change_rate": _round(_change_rate(_ratio_bins(iois))),
        "attack_size_change_rate": _round(_change_rate(group_sizes)),
    }


def build_material_development_profile(
    feature_sets: list[dict[str, float]], *, source_count: int
) -> dict[str, Any]:
    if not feature_sets:
        raise ValueError("at least one material development feature set is required")
    metrics: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = [float(features[metric]) for features in feature_sets]
        metrics[metric] = {
            "p25": _round(_percentile(values, 0.25)),
            "median": _round(_percentile(values, 0.5)),
            "p75": _round(_percentile(values, 0.75)),
        }
    return {
        "schema_version": 1,
        "source_count": source_count,
        "window_count": len(feature_sets),
        "metrics": metrics,
    }


def _material_events(material: Material) -> list[DevelopmentEvent]:
    return [DevelopmentEvent(note.at_ms, note.duration_ms) for note in material.notes]


def evaluate_material_development(
    materials: tuple[Material, ...], profile: dict[str, Any]
) -> dict[str, Any]:
    """三指標が同時に参照下位四分位へ入る素材だけを極端とする。"""
    reports: list[dict[str, Any]] = []
    for material in materials:
        features = extract_material_development(
            _material_events(material), span_ms=material.duration_ms
        )
        lower = [
            metric
            for metric in METRICS
            if features[metric] <= float(profile["metrics"][metric]["p25"])
        ]
        reports.append(
            {
                "material_id": material.material_id,
                "features": features,
                "lower_quartile_metrics": lower,
                "extreme_uniformity": len(lower) == len(METRICS),
            }
        )
    extreme = [report for report in reports if report["extreme_uniformity"]]
    ranked = sorted(
        reports,
        key=lambda report: (
            -len(report["lower_quartile_metrics"]),
            sum(float(report["features"][metric]) for metric in METRICS),
            str(report["material_id"]),
        ),
    )
    return {
        "status": "pass" if reports else "unable_to_investigate",
        "extreme_material_count": len(extreme),
        "worst_material_id": ranked[0]["material_id"] if ranked else None,
        "materials": reports,
    }


def _window_positions(total_span_ms: int, window_ms: int) -> list[int]:
    last = total_span_ms - window_ms
    return sorted({0, last // 2, last}) if last >= 0 else []


def _window_events(
    notes: list[SmfNote], *, start_ms: int, window_ms: int
) -> list[DevelopmentEvent]:
    end_ms = start_ms + window_ms
    return [
        DevelopmentEvent(note.onset_ms - start_ms, note.duration_ms)
        for note in notes
        if start_ms <= note.onset_ms < end_ms
    ]


def extract_reference_windows(notes: list[SmfNote]) -> list[dict[str, float]]:
    if not notes:
        return []
    total_span_ms = max(note.onset_ms + note.duration_ms for note in notes)
    features: list[dict[str, float]] = []
    for window_ms in REFERENCE_WINDOW_LENGTHS_MS:
        for start_ms in _window_positions(total_span_ms, window_ms):
            events = _window_events(notes, start_ms=start_ms, window_ms=window_ms)
            if len(_attack_groups(events, span_ms=window_ms)) < 4:
                continue
            features.append(extract_material_development(events, span_ms=window_ms))
    return features


def build_material_development_reference_profile_from_directory(
    input_dir: Path, *, excluded_names: frozenset[str] = frozenset()
) -> dict[str, Any]:
    feature_sets: list[dict[str, float]] = []
    source_files: list[str] = []
    insufficient_window_files: list[str] = []
    unable: list[dict[str, str]] = []
    excluded_casefold = {name.casefold() for name in excluded_names}
    examined_count = 0
    for path in sorted(Path(input_dir).glob("*.mid"), key=lambda item: item.name.casefold()):
        if path.name.casefold() in excluded_casefold:
            continue
        examined_count += 1
        try:
            windows = extract_reference_windows(load_smf_notes(path))
        except (OSError, ValueError, EOFError) as error:
            unable.append({"name": path.name, "error": f"{type(error).__name__}: {error}"})
            continue
        if not windows:
            insufficient_window_files.append(path.name)
            continue
        feature_sets.extend(windows)
        source_files.append(path.name)
    profile = build_material_development_profile(feature_sets, source_count=len(source_files))
    profile["examined_source_count"] = examined_count
    profile["source_files"] = source_files
    profile["excluded_files"] = sorted(excluded_names)
    profile["insufficient_window_files"] = insufficient_window_files
    profile["unable_to_investigate"] = unable
    return profile
