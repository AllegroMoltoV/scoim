"""終盤と直前区間の差を、拍を仮定せず評価する。"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any

from llm_musical_composer.pilot_features import NoteEvent

COMPARISON_METRICS = (
    "onset_density_ratio",
    "duration_median_ratio",
    "velocity_median_difference",
    "maximum_polyphony_difference",
    "pitch_range_difference",
    "final_sonority_duration_ratio",
)


def _round(value: float) -> float:
    return round(float(value), 8)


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return _round(numerator / denominator)


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return _round(left - right)


def _maximum_polyphony(notes: list[NoteEvent], start: float, end: float) -> int | None:
    changes: list[tuple[float, int]] = []
    for note in notes:
        note_end = note.onset_ms + note.duration_ms
        clipped_start = max(float(note.onset_ms), start)
        clipped_end = min(float(note_end), end)
        if clipped_end <= clipped_start:
            continue
        changes.extend(((clipped_start, 1), (clipped_end, -1)))
    if not changes:
        return None
    current = 0
    maximum = 0
    for _, delta in sorted(changes, key=lambda item: (item[0], item[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


def _segment(notes: list[NoteEvent], start: float, end: float) -> dict[str, float | None]:
    onset_notes = [note for note in notes if start <= note.onset_ms < end]
    pitches = [note.pitch for note in onset_notes]
    return {
        "onset_count": float(len(onset_notes)),
        "duration_median": _median([float(note.duration_ms) for note in onset_notes]),
        "velocity_median": _median([float(note.velocity) for note in onset_notes]),
        "maximum_polyphony": _maximum_polyphony(notes, start, end),
        "pitch_range": float(max(pitches) - min(pitches)) if pitches else None,
    }


def extract_closure_features(
    notes: list[NoteEvent],
    *,
    target_interval: tuple[float, float] = (0.85, 1.0),
    smf_end_ms: int | None = None,
) -> dict[str, Any]:
    """対象区間を同じ長さの直前区間と比較する。"""
    usable = [note for note in notes if note.duration_ms > 0]
    if not usable:
        return {
            "status": "unable_to_investigate",
            "detail": "no complete pitched notes",
            "comparison_metrics": {},
            "diagnostics": {},
        }
    target_start, target_end = target_interval
    width = target_end - target_start
    if width <= 0 or target_start - width < 0 or target_end > 1:
        raise ValueError("target interval must have an equally sized preceding interval")
    sounding_start = min(note.onset_ms for note in usable)
    sounding_end = max(note.onset_ms + note.duration_ms for note in usable)
    span = float(sounding_end - sounding_start)
    if span <= 0:
        return {
            "status": "unable_to_investigate",
            "detail": "sounding duration is zero",
            "comparison_metrics": {},
            "diagnostics": {},
        }

    def position(fraction: float) -> float:
        return sounding_start + span * fraction

    previous = _segment(usable, position(target_start - width), position(target_start))
    target = _segment(usable, position(target_start), position(target_end) + 1e-9)
    target_notes = [
        note for note in usable if position(target_start) <= note.onset_ms <= position(target_end)
    ]
    latest_onset = max((note.onset_ms for note in target_notes), default=None)
    final_sonority = (
        max(
            note.onset_ms + note.duration_ms - latest_onset
            for note in target_notes
            if note.onset_ms == latest_onset
        )
        if latest_onset is not None
        else None
    )
    metrics = {
        "onset_density_ratio": _ratio(target["onset_count"], previous["onset_count"]),
        "duration_median_ratio": _ratio(target["duration_median"], previous["duration_median"]),
        "velocity_median_difference": _difference(
            target["velocity_median"], previous["velocity_median"]
        ),
        "maximum_polyphony_difference": _difference(
            target["maximum_polyphony"], previous["maximum_polyphony"]
        ),
        "pitch_range_difference": _difference(target["pitch_range"], previous["pitch_range"]),
        "final_sonority_duration_ratio": _round(final_sonority / span)
        if final_sonority is not None
        else None,
    }
    end_ms = sounding_end if smf_end_ms is None else max(smf_end_ms, sounding_end)
    return {
        "status": "pass"
        if any(value is not None for value in metrics.values())
        else "unable_to_investigate",
        "target_interval": [target_start, target_end],
        "previous": previous,
        "target": target,
        "comparison_metrics": metrics,
        "diagnostics": {
            "terminal_silence_ratio": _round((end_ms - sounding_end) / span),
            "sounding_note_count": len(usable),
        },
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_closure_profile(feature_sets: list[dict[str, Any]]) -> dict[str, Any]:
    if not feature_sets:
        raise ValueError("at least one closure feature set is required")
    metrics: dict[str, Any] = {}
    for metric in COMPARISON_METRICS:
        values = [
            float(features["comparison_metrics"][metric])
            for features in feature_sets
            if features.get("comparison_metrics", {}).get(metric) is not None
        ]
        metrics[metric] = (
            {
                "available_count": len(values),
                "p25": _round(_percentile(values, 0.25)),
                "median": _round(_percentile(values, 0.5)),
                "p75": _round(_percentile(values, 0.75)),
            }
            if values
            else {"available_count": 0, "status": "unable_to_investigate"}
        )
    return {"schema_version": 1, "source_count": len(feature_sets), "metrics": metrics}


def _distance(value: float | None, bounds: dict[str, Any]) -> float | None:
    if value is None or bounds.get("available_count", 0) == 0:
        return None
    iqr = bounds["p75"] - bounds["p25"]
    scale = iqr if iqr > 1e-9 else max(abs(bounds["median"]), 1.0)
    if value < bounds["p25"]:
        return _round((bounds["p25"] - value) / scale)
    if value > bounds["p75"]:
        return _round((value - bounds["p75"]) / scale)
    return 0.0


def evaluate_closure_against_profile(
    features: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for metric, bounds in profile["metrics"].items():
        value = features.get("comparison_metrics", {}).get(metric)
        results[metric] = {
            "value": value,
            "normalized_distance": _distance(value, bounds),
        }
    return {"metrics": results}


def _bootstrap_interval(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    means = [statistics.mean(rng.choice(values) for _ in values) for _ in range(max(1, samples))]
    return (_round(_percentile(means, 0.025)), _round(_percentile(means, 0.975)))


def evaluate_ending_discrimination(
    corpus: list[list[NoteEvent]], *, bootstrap_samples: int = 2_000, seed: int = 0
) -> dict[str, Any]:
    """実終盤と 50% から 65% の疑似終盤を leave-one-out で比較する。"""
    usable: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for notes in corpus:
        actual = extract_closure_features(notes)
        pseudo = extract_closure_features(notes, target_interval=(0.5, 0.65))
        if actual["status"] == "pass" and pseudo["status"] == "pass":
            usable.append((actual, pseudo))
    if len(usable) < 3:
        return {
            "status": "unable_to_investigate",
            "available_count": len(usable),
            "detail": "at least three usable songs are required",
        }
    outcomes: list[float] = []
    comparisons: list[dict[str, Any]] = []
    for index, (actual, pseudo) in enumerate(usable):
        profile = build_closure_profile(
            [
                other_actual
                for other_index, (other_actual, _) in enumerate(usable)
                if other_index != index
            ]
        )
        actual_evaluation = evaluate_closure_against_profile(actual, profile)["metrics"]
        pseudo_evaluation = evaluate_closure_against_profile(pseudo, profile)["metrics"]
        actual_better = 0
        pseudo_better = 0
        available = 0
        metric_comparisons: dict[str, Any] = {}
        for metric in COMPARISON_METRICS:
            actual_distance = actual_evaluation[metric]["normalized_distance"]
            pseudo_distance = pseudo_evaluation[metric]["normalized_distance"]
            if actual_distance is None or pseudo_distance is None:
                metric_comparisons[metric] = {
                    "actual_distance": actual_distance,
                    "pseudo_distance": pseudo_distance,
                    "preferred": "unable_to_investigate",
                }
                continue
            available += 1
            preferred = "tie"
            if actual_distance < pseudo_distance:
                actual_better += 1
                preferred = "actual"
            elif pseudo_distance < actual_distance:
                pseudo_better += 1
                preferred = "pseudo"
            metric_comparisons[metric] = {
                "actual_distance": actual_distance,
                "pseudo_distance": pseudo_distance,
                "preferred": preferred,
            }
        required_better_metrics = max(2, math.ceil(available / 3))
        outcome = (
            1.0
            if available
            and actual_better > pseudo_better
            and actual_better >= required_better_metrics
            else 0.0
        )
        outcomes.append(outcome)
        comparisons.append(
            {
                "song_index": index,
                "available_metrics": available,
                "actual_better_metrics": actual_better,
                "pseudo_better_metrics": pseudo_better,
                "required_better_metrics": required_better_metrics,
                "actual_wins": bool(outcome),
                "metric_comparisons": metric_comparisons,
            }
        )
    lower, upper = _bootstrap_interval(outcomes, bootstrap_samples, seed)
    metric_win_counts = {
        metric: {
            "actual_better": sum(
                comparison["metric_comparisons"][metric]["preferred"] == "actual"
                for comparison in comparisons
            ),
            "pseudo_better": sum(
                comparison["metric_comparisons"][metric]["preferred"] == "pseudo"
                for comparison in comparisons
            ),
            "tie": sum(
                comparison["metric_comparisons"][metric]["preferred"] == "tie"
                for comparison in comparisons
            ),
            "unable_to_investigate": sum(
                comparison["metric_comparisons"][metric]["preferred"] == "unable_to_investigate"
                for comparison in comparisons
            ),
        }
        for metric in COMPARISON_METRICS
    }
    return {
        "status": "pass" if lower > 0.5 else "fail",
        "available_count": len(usable),
        "actual_win_rate": _round(statistics.mean(outcomes)),
        "confidence_interval_95": {"lower": lower, "upper": upper},
        "metric_win_counts": metric_win_counts,
        "comparisons": comparisons,
    }


def run_closure_controls(notes: list[NoteEvent]) -> dict[str, dict[str, str]]:
    """末尾無音だけでは比較指標が改善しないことを確認する。"""
    if not notes:
        return {
            "terminal_silence_only": {
                "status": "unable_to_investigate",
                "detail": "no complete pitched notes",
            }
        }
    sounding_end = max(note.onset_ms + note.duration_ms for note in notes)
    original = extract_closure_features(notes, smf_end_ms=sounding_end)
    silent = extract_closure_features(notes, smf_end_ms=sounding_end + 5_000)
    passed = (
        original["comparison_metrics"] == silent["comparison_metrics"]
        and original["diagnostics"]["terminal_silence_ratio"]
        < silent["diagnostics"]["terminal_silence_ratio"]
    )
    return {
        "terminal_silence_only": {
            "status": "pass" if passed else "fail",
            "detail": "silence changed diagnostics only"
            if passed
            else "silence changed a comparison metric",
        }
    }
