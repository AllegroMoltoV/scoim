"""楽譜位置と共有演奏時間を組にした有限候補を生成する。"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from typing import Any

from llm_musical_composer.reference_decomposition import (
    ObservedNote,
    ObservedPerformance,
)
from llm_musical_composer.score_timing_oracle_diagnostics import (
    fit_piecewise_time_map,
)


@dataclass(frozen=True)
class GroupingAttack:
    onset_us: int
    source_event_ids: tuple[str, ...]
    relative_onsets_us: tuple[int, ...]


@dataclass(frozen=True)
class HypothesisAttackGroupRef:
    group_index: int
    score_position: int


@dataclass(frozen=True)
class SharedTimeMapKnot:
    score_position: int
    observed_time_us: int


@dataclass(frozen=True)
class LocalCoordination:
    group_index: int
    relative_onsets_us: tuple[int, ...]


@dataclass(frozen=True)
class ScoreTimingHypothesisV0:
    candidate_id: str
    status: str
    search_status: str
    source_ledger_sha256: str
    grouping_profile_id: str
    score_grid_id: str
    requested_segment_count: int
    actual_segment_count: int
    attack_group_refs: tuple[HypothesisAttackGroupRef, ...]
    shared_time_map: tuple[SharedTimeMapKnot, ...]
    local_coordination: tuple[LocalCoordination, ...]
    equivalence: tuple[str, ...]
    assumptions: tuple[str, ...]
    evidence_event_ids: tuple[str, ...]
    metrics: dict[str, int | float]


@dataclass(frozen=True)
class ScoreTimingNegativeControl:
    control_id: str
    status: str
    score_positions: tuple[int, ...]
    metrics: dict[str, int | float]


@dataclass(frozen=True)
class ScoreTimingCandidateSet:
    status: str
    reason: str
    candidates: tuple[ScoreTimingHypothesisV0, ...]
    negative_controls: tuple[ScoreTimingNegativeControl, ...]


@dataclass(frozen=True)
class IntervalVocabularyFit:
    intervals: tuple[Fraction, ...]
    normalized_intervals: tuple[int, ...]
    scale: Fraction
    squared_error: Fraction
    search_status: str


@dataclass(frozen=True)
class DynamicScoreTimingFit:
    intervals: tuple[Fraction, ...]
    normalized_intervals: tuple[int, ...]
    boundaries: tuple[int, ...]
    boundary_history: tuple[tuple[int, ...], ...]
    search_status: str
    iteration_count: int


VOCABULARIES: dict[str, tuple[Fraction, ...]] = {
    "binary": tuple(Fraction(value) for value in (1, 2, 4, 8)),
    "binary-dotted": tuple(Fraction(value, 2) for value in (2, 3, 4, 6, 8, 12, 16)),
    "binary-dotted-triplet": tuple(
        sorted(
            {
                Fraction(2, 3),
                Fraction(1),
                Fraction(4, 3),
                Fraction(3, 2),
                Fraction(2),
                Fraction(8, 3),
                Fraction(3),
                Fraction(4),
                Fraction(16, 3),
                Fraction(6),
                Fraction(8),
            }
        )
    ),
}


def _ordered_notes(performance: ObservedPerformance) -> tuple[ObservedNote, ...]:
    return tuple(
        sorted(
            performance.notes,
            key=lambda note: (note.onset_us, note.pitch, note.note_on_event_id),
        )
    )


def _group_with_window(
    notes: tuple[ObservedNote, ...], window_us: int
) -> tuple[GroupingAttack, ...]:
    groups: list[GroupingAttack] = []
    index = 0
    while index < len(notes):
        anchor = notes[index].onset_us
        members: list[ObservedNote] = []
        while index < len(notes) and notes[index].onset_us - anchor <= window_us:
            members.append(notes[index])
            index += 1
        groups.append(
            GroupingAttack(
                onset_us=anchor,
                source_event_ids=tuple(note.note_on_event_id for note in members),
                relative_onsets_us=tuple(note.onset_us - anchor for note in members),
            )
        )
    return tuple(groups)


def _rolled_merge_groups(
    base: tuple[GroupingAttack, ...], notes_by_id: dict[str, ObservedNote]
) -> tuple[GroupingAttack, ...]:
    merged: list[GroupingAttack] = []
    index = 0
    while index < len(base):
        anchor = base[index].onset_us
        event_ids = list(base[index].source_event_ids)
        current_pitches = [notes_by_id[event_id].pitch for event_id in event_ids]
        index += 1
        while index < len(base) and base[index].onset_us - anchor <= 60_000:
            next_ids = base[index].source_event_ids
            next_pitches = [notes_by_id[event_id].pitch for event_id in next_ids]
            if max(current_pitches) >= min(next_pitches):
                break
            event_ids.extend(next_ids)
            current_pitches.extend(next_pitches)
            index += 1
        ordered = sorted(
            (notes_by_id[event_id] for event_id in event_ids),
            key=lambda note: (note.onset_us, note.pitch, note.note_on_event_id),
        )
        merged.append(
            GroupingAttack(
                onset_us=anchor,
                source_event_ids=tuple(note.note_on_event_id for note in ordered),
                relative_onsets_us=tuple(note.onset_us - anchor for note in ordered),
            )
        )
    return tuple(merged)


def build_grouping_profiles(
    performance: ObservedPerformance,
) -> dict[str, tuple[GroupingAttack, ...]]:
    """30 ms群と、低音から高音への60 ms rolled候補を分けて返す。"""
    notes = _ordered_notes(performance)
    notes_by_id = {note.note_on_event_id: note for note in notes}
    base = _group_with_window(notes, 30_000)
    return {
        "attack-30ms": base,
        "rolled-merge-60ms": _rolled_merge_groups(base, notes_by_id),
        "rolled-separate": _group_with_window(notes, 0),
    }


def _percentile(values: tuple[int, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _fit_segment_local_v0(
    observed_intervals: tuple[int, ...], vocabulary: tuple[Fraction, ...]
) -> tuple[tuple[Fraction, ...], str]:
    quantiles = tuple(
        _percentile(observed_intervals, fraction) for fraction in (0.1, 0.25, 0.5, 0.75, 0.9)
    )
    seed_tokens = vocabulary[: min(4, len(vocabulary))]
    seeds = sorted(
        {value / float(token) for value in quantiles for token in seed_tokens if value > 0}
    )
    completed: list[tuple[float, tuple[Fraction, ...], str]] = []
    for initial_scale in seeds:
        scale = initial_scale
        visited: set[tuple[Fraction, ...]] = set()
        previous: tuple[Fraction, ...] | None = None
        while True:
            state = tuple(
                min(
                    vocabulary,
                    key=lambda token: (
                        abs(observed - scale * float(token)),
                        token,
                    ),
                )
                for observed in observed_intervals
            )
            if state == previous:
                search_status = "complete"
                break
            if state in visited:
                search_status = "not_converged"
                break
            visited.add(state)
            previous = state
            denominator = sum(float(token) ** 2 for token in state)
            scale = (
                sum(
                    observed * float(token)
                    for observed, token in zip(observed_intervals, state, strict=True)
                )
                / denominator
            )
        error = sum(
            (observed - scale * float(token)) ** 2
            for observed, token in zip(observed_intervals, state, strict=True)
        )
        completed.append((error, state, search_status))
    _, best_state, best_status = min(
        completed,
        key=lambda item: (
            item[0],
            tuple((token.numerator, token.denominator) for token in item[1]),
            item[2],
        ),
    )
    return best_state, best_status


def fit_interval_vocabulary_global(
    observed_intervals: tuple[int, ...],
    vocabulary: tuple[Fraction, ...],
) -> IntervalVocabularyFit:
    """尺度の切替点を走査し、固定区分の音価列を大域最小化する。"""
    if not observed_intervals or any(value <= 0 for value in observed_intervals):
        raise ValueError("observed intervals must be positive and non-empty")
    vocabulary = tuple(sorted(set(vocabulary)))
    if not vocabulary or any(value <= 0 for value in vocabulary):
        raise ValueError("vocabulary values must be positive and non-empty")
    events = sorted(
        (
            Fraction(2 * observed, lower + upper),
            observed,
            lower,
            upper,
        )
        for observed in observed_intervals
        for lower, upper in pairwise(vocabulary)
    )
    maximum = vocabulary[-1]
    coefficient_squared = Fraction(len(observed_intervals)) * maximum * maximum
    coefficient_linear = maximum * sum(observed_intervals)
    coefficient_constant = sum(value * value for value in observed_intervals)
    best_error: Fraction | None = None
    best_scales: set[Fraction] = set()

    def consider(lower_scale: Fraction, upper_scale: Fraction | None) -> None:
        nonlocal best_error, best_scales
        scale = coefficient_linear / coefficient_squared
        if scale < lower_scale:
            scale = lower_scale
        if upper_scale is not None and scale > upper_scale:
            scale = upper_scale
        squared_error = (
            coefficient_constant
            - 2 * scale * coefficient_linear
            + scale * scale * coefficient_squared
        )
        if best_error is None or squared_error < best_error:
            best_error = squared_error
            best_scales = {scale}
        elif squared_error == best_error:
            best_scales.add(scale)

    lower_scale = Fraction(0)
    index = 0
    while index < len(events):
        boundary = events[index][0]
        consider(lower_scale, boundary)
        while index < len(events) and events[index][0] == boundary:
            _, observed, lower, upper = events[index]
            coefficient_linear += Fraction(observed) * (lower - upper)
            coefficient_squared += lower * lower - upper * upper
            index += 1
        lower_scale = boundary
    consider(lower_scale, None)
    assert best_error is not None and best_scales
    fits = []
    for scale in best_scales:
        intervals = tuple(
            min(
                vocabulary,
                key=lambda token: (abs(Fraction(observed) - scale * token), token),
            )
            for observed in observed_intervals
        )
        normalized = _integer_intervals(intervals)
        squared_error = sum(
            (Fraction(observed) - scale * interval) ** 2
            for observed, interval in zip(observed_intervals, intervals, strict=True)
        )
        fits.append(
            IntervalVocabularyFit(
                intervals=intervals,
                normalized_intervals=normalized,
                scale=scale,
                squared_error=squared_error,
                search_status="global_optimum",
            )
        )
    return min(
        fits,
        key=lambda fit: (
            fit.squared_error,
            fit.normalized_intervals,
            fit.scale,
            fit.intervals,
        ),
    )


def _fit_segment(
    observed_intervals: tuple[int, ...], vocabulary: tuple[Fraction, ...]
) -> tuple[tuple[Fraction, ...], str]:
    fit = fit_interval_vocabulary_global(observed_intervals, vocabulary)
    return fit.intervals, "complete"


def _segment_boundaries(interval_count: int, requested_count: int) -> tuple[int, ...]:
    actual = min(interval_count, requested_count)
    return (
        *(index * interval_count // actual for index in range(actual)),
        interval_count,
    )


def _integer_intervals(values: tuple[Fraction, ...]) -> tuple[int, ...]:
    denominator = math.lcm(*(value.denominator for value in values))
    integers = tuple(value.numerator * denominator // value.denominator for value in values)
    divisor = math.gcd(*integers)
    return tuple(value // divisor for value in integers)


def _score_positions(intervals: tuple[int, ...]) -> tuple[int, ...]:
    positions = [0]
    for interval in intervals:
        positions.append(positions[-1] + interval)
    return tuple(positions)


def _classify_boundary_transition(
    history: tuple[tuple[int, ...], ...],
    next_boundaries: tuple[int, ...],
) -> str:
    """直前一致を固定点、それ以前への回帰を循環として区別する。"""
    if not history:
        raise ValueError("boundary history must be non-empty")
    if next_boundaries == history[-1]:
        return "complete"
    if next_boundaries in history[:-1]:
        return "not_converged"
    return "continue"


def fit_dynamic_score_timing(
    observed_intervals: tuple[int, ...],
    vocabulary: tuple[Fraction, ...],
    *,
    requested_segment_count: int,
) -> DynamicScoreTimingFit:
    """区分内音価と疎な時間写像境界を固定点または循環まで交互更新する。"""
    if len(observed_intervals) < 2 or any(value <= 0 for value in observed_intervals):
        raise ValueError("at least two positive observed intervals are required")
    if requested_segment_count < 1 or requested_segment_count > 8:
        raise ValueError("requested segment count must be between 1 and 8")
    actual_count = min(requested_segment_count, len(observed_intervals) // 2)
    boundaries = (
        *(index * len(observed_intervals) // actual_count for index in range(actual_count)),
        len(observed_intervals),
    )
    times = _score_positions(observed_intervals)
    history: list[tuple[int, ...]] = []
    iteration_count = 0
    while True:
        history.append(boundaries)
        iteration_count += 1
        fraction_intervals = tuple(
            interval
            for start, end in pairwise(boundaries)
            for interval in fit_interval_vocabulary_global(
                observed_intervals[start:end],
                vocabulary,
            ).intervals
        )
        normalized_intervals = _integer_intervals(fraction_intervals)
        positions = _score_positions(normalized_intervals)
        next_boundaries = fit_piecewise_time_map(
            positions,
            times,
            requested_segment_count=requested_segment_count,
        ).boundaries
        transition = _classify_boundary_transition(tuple(history), next_boundaries)
        if transition != "continue":
            return DynamicScoreTimingFit(
                intervals=fraction_intervals,
                normalized_intervals=normalized_intervals,
                boundaries=boundaries,
                boundary_history=tuple(history),
                search_status=transition,
                iteration_count=iteration_count,
            )
        boundaries = next_boundaries


def _reconstruct_group_times(
    positions: tuple[int, ...],
    times: tuple[int, ...],
    boundaries: tuple[int, ...],
) -> tuple[int, ...]:
    reconstructed = [0] * len(times)
    for start, end in pairwise(boundaries):
        score_span = positions[end] - positions[start]
        time_span = times[end] - times[start]
        for index in range(start, end + 1):
            offset = Fraction(
                (positions[index] - positions[start]) * time_span,
                score_span,
            )
            reconstructed[index] = times[start] + round(offset)
    return tuple(reconstructed)


def _linear_fit(
    positions: tuple[int, ...], times: tuple[int, ...], indices: tuple[int, ...]
) -> tuple[float, float]:
    x_values = [positions[index] for index in indices]
    y_values = [times[index] for index in indices]
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0:
        return y_mean, 0.0
    slope = (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        / denominator
    )
    return y_mean - slope * x_mean, slope


def _in_sample_interpolation_errors(
    positions: tuple[int, ...],
    times: tuple[int, ...],
    boundaries: tuple[int, ...],
) -> tuple[int, ...]:
    errors: list[int] = []
    for start, end in pairwise(boundaries):
        interior = tuple(range(start + 1, end))
        withheld = interior[::2]
        if not withheld:
            continue
        withheld_set = set(withheld)
        kept = tuple(index for index in range(start, end + 1) if index not in withheld_set)
        intercept, slope = _linear_fit(positions, times, kept)
        errors.extend(
            abs(times[index] - round(intercept + slope * positions[index])) for index in withheld
        )
    return tuple(errors)


def _candidate_metrics(
    groups: tuple[GroupingAttack, ...],
    positions: tuple[int, ...],
    reconstructed: tuple[int, ...],
    boundaries: tuple[int, ...],
    vocabulary_size: int,
) -> dict[str, int | float]:
    times = tuple(group.onset_us for group in groups)
    errors = tuple(
        abs(observed - predicted) for observed, predicted in zip(times, reconstructed, strict=True)
    )
    interpolation = _in_sample_interpolation_errors(positions, times, boundaries)
    return {
        "mean_group_anchor_error_us": round(statistics.fmean(errors), 3),
        "maximum_group_anchor_error_us": max(errors),
        "in_sample_interpolation_mean_error_us": (
            round(statistics.fmean(interpolation), 3) if interpolation else 0.0
        ),
        "in_sample_interpolation_maximum_error_us": max(interpolation, default=0),
        "vocabulary_size": vocabulary_size,
        "time_map_knot_count": len(boundaries),
        "local_coordination_group_count": sum(
            any(offset != 0 for offset in group.relative_onsets_us) for group in groups
        ),
    }


def _normalized_serialization_bytes(
    groups: tuple[GroupingAttack, ...],
    positions: tuple[int, ...],
    boundaries: tuple[int, ...],
) -> int:
    serialized = {
        "refs": list(enumerate(positions)),
        "knots": [(positions[index], groups[index].onset_us) for index in boundaries],
        "local": [
            (index, group.relative_onsets_us)
            for index, group in enumerate(groups)
            if any(offset != 0 for offset in group.relative_onsets_us)
        ],
    }
    return len(json.dumps(serialized, separators=(",", ":")).encode("utf-8"))


def _build_candidate(
    *,
    source_ledger_sha256: str,
    grouping_profile_id: str,
    groups: tuple[GroupingAttack, ...],
    score_grid_id: str,
    vocabulary: tuple[Fraction, ...],
    requested_segment_count: int,
    boundary_mode: str = "fixed",
) -> ScoreTimingHypothesisV0:
    times = tuple(group.onset_us for group in groups)
    observed_intervals = tuple(right - left for left, right in pairwise(times))
    if boundary_mode == "fixed":
        boundaries = _segment_boundaries(len(observed_intervals), requested_segment_count)
        fraction_intervals: list[Fraction] = []
        statuses: list[str] = []
        for start, end in pairwise(boundaries):
            fitted, status = _fit_segment(observed_intervals[start:end], vocabulary)
            fraction_intervals.extend(fitted)
            statuses.append(status)
        fitted_intervals = tuple(fraction_intervals)
        search_status = (
            "complete" if all(status == "complete" for status in statuses) else "not_converged"
        )
        boundary_assumption = "time-map segment boundaries are fixed by attack interval index"
        candidate_suffix = ""
    elif boundary_mode == "dynamic":
        dynamic_fit = fit_dynamic_score_timing(
            observed_intervals,
            vocabulary,
            requested_segment_count=requested_segment_count,
        )
        boundaries = dynamic_fit.boundaries
        fitted_intervals = dynamic_fit.intervals
        search_status = dynamic_fit.search_status
        boundary_assumption = (
            "time-map segment boundaries alternate with globally fitted interval vocabulary"
        )
        candidate_suffix = "--dynamic-boundaries"
    else:
        raise ValueError(f"unsupported boundary mode: {boundary_mode}")
    intervals = _integer_intervals(fitted_intervals)
    positions = _score_positions(intervals)
    reconstructed = _reconstruct_group_times(positions, times, boundaries)
    refs = tuple(
        HypothesisAttackGroupRef(index, position) for index, position in enumerate(positions)
    )
    knots = tuple(SharedTimeMapKnot(positions[index], times[index]) for index in boundaries)
    local = tuple(
        LocalCoordination(index, group.relative_onsets_us)
        for index, group in enumerate(groups)
        if any(offset != 0 for offset in group.relative_onsets_us)
    )
    metrics = _candidate_metrics(
        groups,
        positions,
        reconstructed,
        boundaries,
        len(set(intervals)),
    )
    metrics["normalized_serialization_bytes"] = _normalized_serialization_bytes(
        groups,
        positions,
        boundaries,
    )
    candidate_id = (
        f"{grouping_profile_id}--{score_grid_id}--segments-{requested_segment_count}"
        f"{candidate_suffix}"
    )
    evidence = tuple(
        event_id
        for group in (groups[0], groups[-1])
        for event_id in (group.source_event_ids[0], group.source_event_ids[-1])
    )
    return ScoreTimingHypothesisV0(
        candidate_id=candidate_id,
        status="candidate",
        search_status=search_status,
        source_ledger_sha256=source_ledger_sha256,
        grouping_profile_id=grouping_profile_id,
        score_grid_id=score_grid_id,
        requested_segment_count=requested_segment_count,
        actual_segment_count=len(boundaries) - 1,
        attack_group_refs=refs,
        shared_time_map=knots,
        local_coordination=local,
        equivalence=("global positive score scale is normalized by interval gcd",),
        assumptions=(
            "only note-on group anchors are modeled",
            "meter and declared BPM are not used",
            boundary_assumption,
        ),
        evidence_event_ids=evidence,
        metrics=metrics,
    )


def _negative_controls(
    groups: tuple[GroupingAttack, ...],
) -> tuple[ScoreTimingNegativeControl, ...]:
    times = tuple(group.onset_us for group in groups)
    observed_intervals = tuple(right - left for left, right in pairwise(times))
    observed_positions = _score_positions(
        _integer_intervals(tuple(map(Fraction, observed_intervals)))
    )
    uniform_positions = tuple(range(len(groups)))
    boundaries = (0, len(groups) - 1)
    controls = []
    for control_id, positions in (
        ("observed-copy-negative-control", observed_positions),
        ("uniform-grid-baseline", uniform_positions),
    ):
        reconstructed = _reconstruct_group_times(positions, times, boundaries)
        metrics = _candidate_metrics(
            groups,
            positions,
            reconstructed,
            boundaries,
            len(set(right - left for left, right in pairwise(positions))),
        )
        metrics["normalized_serialization_bytes"] = _normalized_serialization_bytes(
            groups,
            positions,
            boundaries,
        )
        controls.append(
            ScoreTimingNegativeControl(
                control_id=control_id,
                status="negative_control",
                score_positions=positions,
                metrics=metrics,
            )
        )
    return tuple(controls)


def generate_score_timing_candidates(
    performance: ObservedPerformance, *, source_ledger_sha256: str
) -> ScoreTimingCandidateSet:
    """固定した27候補と二つの負対照を生成する。"""
    paired_note_on_ids = {note.note_on_event_id for note in performance.notes}
    observed_note_on_ids = {
        event_id for group in performance.attack_groups for event_id in group.note_on_event_ids
    }
    if paired_note_on_ids != observed_note_on_ids:
        return ScoreTimingCandidateSet(
            status="unable_to_investigate",
            reason="note matching does not preserve every observed note-on",
            candidates=(),
            negative_controls=(),
        )
    profiles = build_grouping_profiles(performance)
    if any(len(groups) < 2 for groups in profiles.values()):
        return ScoreTimingCandidateSet(
            status="outside_candidate_space",
            reason="at least two attack groups are required",
            candidates=(),
            negative_controls=(),
        )
    candidates = tuple(
        _build_candidate(
            source_ledger_sha256=source_ledger_sha256,
            grouping_profile_id=grouping_profile_id,
            groups=groups,
            score_grid_id=score_grid_id,
            vocabulary=vocabulary,
            requested_segment_count=segment_count,
        )
        for grouping_profile_id, groups in profiles.items()
        for score_grid_id, vocabulary in VOCABULARIES.items()
        for segment_count in (1, 4, 8)
    )
    return ScoreTimingCandidateSet(
        status="assessed",
        reason="fixed candidate family completed",
        candidates=candidates,
        negative_controls=_negative_controls(profiles["attack-30ms"]),
    )


def generate_dynamic_score_timing_candidates(
    performance: ObservedPerformance, *, source_ledger_sha256: str
) -> ScoreTimingCandidateSet:
    """固定27候補と比較する可変境界18候補を生成する。"""
    paired_note_on_ids = {note.note_on_event_id for note in performance.notes}
    observed_note_on_ids = {
        event_id for group in performance.attack_groups for event_id in group.note_on_event_ids
    }
    if paired_note_on_ids != observed_note_on_ids:
        return ScoreTimingCandidateSet(
            status="unable_to_investigate",
            reason="note matching does not preserve every observed note-on",
            candidates=(),
            negative_controls=(),
        )
    profiles = build_grouping_profiles(performance)
    if any(len(groups) < 3 for groups in profiles.values()):
        return ScoreTimingCandidateSet(
            status="outside_candidate_space",
            reason="at least three attack groups are required for dynamic boundaries",
            candidates=(),
            negative_controls=(),
        )
    unique_profiles: dict[tuple[GroupingAttack, ...], str] = {}
    for profile_id, groups in profiles.items():
        unique_profiles.setdefault(groups, profile_id)
    candidates = tuple(
        _build_candidate(
            source_ledger_sha256=source_ledger_sha256,
            grouping_profile_id=grouping_profile_id,
            groups=groups,
            score_grid_id=score_grid_id,
            vocabulary=vocabulary,
            requested_segment_count=segment_count,
            boundary_mode="dynamic",
        )
        for groups, grouping_profile_id in unique_profiles.items()
        for score_grid_id, vocabulary in VOCABULARIES.items()
        for segment_count in (4, 8)
    )
    return ScoreTimingCandidateSet(
        status="assessed",
        reason="dynamic boundary candidate family completed after exact grouping deduplication",
        candidates=candidates,
        negative_controls=(),
    )


def _normalized_position_intervals(positions: tuple[int, ...]) -> tuple[int, ...] | None:
    intervals = tuple(right - left for left, right in pairwise(positions))
    if not intervals or any(interval <= 0 for interval in intervals):
        return None
    divisor = math.gcd(*intervals)
    return tuple(interval // divisor for interval in intervals)


def compare_grouping_to_known_score(
    groups: tuple[GroupingAttack, ...],
    known_score_units_by_event_id: dict[str, int],
) -> dict[str, int | str]:
    """発音群の結合過多、分割過多、証拠欠落を音価比較前に数える。"""
    observed_event_ids = {event_id for group in groups for event_id in group.source_event_ids}
    missing_event_count = len(set(known_score_units_by_event_id) - observed_event_ids)
    unexpected_event_count = len(observed_event_ids - set(known_score_units_by_event_id))
    merged_group_count = 0
    unit_group_counts: dict[int, int] = {}
    for group in groups:
        units = {
            known_score_units_by_event_id[event_id]
            for event_id in group.source_event_ids
            if event_id in known_score_units_by_event_id
        }
        if len(units) > 1:
            merged_group_count += 1
        for unit in units:
            unit_group_counts[unit] = unit_group_counts.get(unit, 0) + 1
    split_position_count = sum(count > 1 for count in unit_group_counts.values())
    if missing_event_count or unexpected_event_count:
        status = "evidence_mismatch"
    elif merged_group_count:
        status = "merges_known_score_positions"
    elif split_position_count:
        status = "splits_known_score_positions"
    else:
        status = "compatible"
    return {
        "status": status,
        "group_count": len(groups),
        "merged_multiple_position_group_count": merged_group_count,
        "split_known_position_count": split_position_count,
        "missing_event_count": missing_event_count,
        "unexpected_event_count": unexpected_event_count,
    }


def compare_candidate_to_known_score(
    candidate: ScoreTimingHypothesisV0,
    groups: tuple[GroupingAttack, ...],
    known_score_units_by_event_id: dict[str, int],
) -> dict[str, Any]:
    """探索後の候補位置比を既知楽譜と全体倍率を除いて比較する。"""
    if len(candidate.attack_group_refs) != len(groups):
        return {"status": "group_count_mismatch"}
    known_positions: list[int] = []
    for group in groups:
        units = {
            known_score_units_by_event_id[event_id]
            for event_id in group.source_event_ids
            if event_id in known_score_units_by_event_id
        }
        if len(units) != 1 or len(units) != len(
            {known_score_units_by_event_id.get(event_id) for event_id in group.source_event_ids}
        ):
            return {"status": "grouping_conflicts_known_score"}
        known_positions.append(next(iter(units)))
    inferred_positions = tuple(item.score_position for item in candidate.attack_group_refs)
    expected_pattern = _normalized_position_intervals(tuple(known_positions))
    inferred_pattern = _normalized_position_intervals(inferred_positions)
    if expected_pattern is None:
        return {"status": "grouping_splits_known_score_position"}
    raw_expected = tuple(right - left for left, right in pairwise(known_positions))
    raw_inferred = tuple(right - left for left, right in pairwise(inferred_positions))
    scale_counts: dict[Fraction, int] = {}
    for inferred, expected in zip(raw_inferred, raw_expected, strict=True):
        scale = Fraction(inferred, expected)
        scale_counts[scale] = scale_counts.get(scale, 0) + 1
    best_scale, matching_count = min(
        scale_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    mismatches = [
        {
            "interval_index": index,
            "expected_interval": expected,
            "inferred_interval": inferred,
        }
        for index, (inferred, expected) in enumerate(zip(raw_inferred, raw_expected, strict=True))
        if Fraction(inferred, expected) != best_scale
    ]
    return {
        "status": "equivalent" if inferred_pattern == expected_pattern else "different",
        "expected_interval_pattern": expected_pattern,
        "inferred_interval_pattern": inferred_pattern,
        "best_scale": [best_scale.numerator, best_scale.denominator],
        "best_scale_matching_interval_count": matching_count,
        "interval_count": len(raw_expected),
        "best_scale_interval_match_rate": round(matching_count / len(raw_expected), 8),
        "mismatched_intervals": mismatches,
    }
