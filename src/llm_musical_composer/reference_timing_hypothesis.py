"""既知楽譜を使い、共有演奏時間と局所打鍵差の同定範囲を調べる。"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise
from statistics import median
from typing import Any

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    ordered_leaf_schedule,
    render_performance,
)
from llm_musical_composer.reference_decomposition import (
    ObservedNote,
    ObservedPerformance,
    ObservedSmf,
)


@dataclass(frozen=True)
class TimingCandidateEquivalence:
    status: str
    candidate_space_id: str
    candidate_space_sha256: str
    surface_candidate_count: int
    semantic_candidate_count: int
    semantic_fingerprints: dict[str, str]
    equivalence_classes: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class MonotonicTimeExplanation:
    score_positions: tuple[int, ...]
    observed_times: tuple[int, ...]
    reconstructed_times: tuple[int, ...]
    local_score_intervals: tuple[int, ...]
    segment_slopes: tuple[Fraction, ...]


@dataclass(frozen=True)
class ExpectedScoreAttack:
    event_id: str
    score_unit: int
    pitch: int
    voice: str


@dataclass(frozen=True)
class AlignedExpectedObservedAttack:
    expected: ExpectedScoreAttack
    observed: ObservedNote


@dataclass(frozen=True)
class KnownScoreNoteAlignment:
    status: str
    matches: tuple[AlignedExpectedObservedAttack, ...]
    assumptions: tuple[str, ...]
    unmatched_expected_count: int
    unmatched_observed_count: int


@dataclass(frozen=True)
class TimingAttackGroupEvidence:
    score_unit: int
    evidence_event_ids: tuple[str, ...]
    minimum_anchor_us: int
    median_anchor_us: int
    minimum_anchor_residuals_us: tuple[int, ...]
    median_anchor_residuals_us: tuple[int, ...]
    invariant_relative_onsets_us: tuple[int, ...]
    invariant_spread_us: int
    coordination_status: str


@dataclass(frozen=True)
class KnownScoreTimingEvidence:
    status: str
    score_known: bool
    note_on_status: str
    note_off_status: str
    cc64_status: str
    attack_groups: tuple[TimingAttackGroupEvidence, ...]
    assumptions: tuple[str, ...]
    unmatched_expected_count: int
    unmatched_observed_count: int


def _stable_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def timing_semantic_fingerprint(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
) -> str:
    """表記ではなく、レンダリング後の時間意味をハッシュ化する。"""
    rendered = render_performance(plan, score, performance)
    payload = {
        "duration_ms": rendered.duration_ms,
        "notes": [
            {
                "event_id": note.event_id,
                "at_ms": note.at_ms,
                "duration_ms": note.duration_ms,
            }
            for note in rendered.notes
        ],
        "pedals": [
            {
                "event_id": pedal.event_id,
                "at_ms": pedal.at_ms,
                "value": pedal.value,
            }
            for pedal in rendered.pedals
        ],
        "node_intervals": rendered.node_intervals,
    }
    return _stable_hash(payload)


def timing_candidate_equivalence(
    plan: PiecePlan,
    score: ScoreSpec,
    candidates: Mapping[str, PerformanceSpec],
    *,
    candidate_space_id: str = "timing-known-equivalence-v1",
) -> TimingCandidateEquivalence:
    """候補表記をレンダリング意味の同値類へまとめる。"""
    if not candidates:
        raise ValueError("timing candidate space must not be empty")
    fingerprints = {
        candidate_id: timing_semantic_fingerprint(plan, score, performance)
        for candidate_id, performance in sorted(candidates.items())
    }
    grouped: dict[str, list[str]] = defaultdict(list)
    for candidate_id, fingerprint in fingerprints.items():
        grouped[fingerprint].append(candidate_id)
    equivalence_classes = {
        fingerprint: tuple(candidate_ids) for fingerprint, candidate_ids in sorted(grouped.items())
    }
    semantic_count = len(equivalence_classes)
    status = (
        "identifiable_within_candidate_space" if semantic_count == 1 else "equivalent_candidates"
    )
    candidate_space_payload = {
        "candidate_space_id": candidate_space_id,
        "candidate_fingerprints": fingerprints,
    }
    return TimingCandidateEquivalence(
        status=status,
        candidate_space_id=candidate_space_id,
        candidate_space_sha256=_stable_hash(candidate_space_payload),
        surface_candidate_count=len(fingerprints),
        semantic_candidate_count=semantic_count,
        semantic_fingerprints=fingerprints,
        equivalence_classes=equivalence_classes,
    )


def monotonic_time_explanation(
    score_positions: Sequence[int],
    observed_times: Sequence[int],
) -> MonotonicTimeExplanation:
    """楽譜位置の各節点を観測時刻へ写す単調な説明を作る。"""
    positions = tuple(int(value) for value in score_positions)
    times = tuple(int(value) for value in observed_times)
    if len(positions) != len(times) or len(positions) < 2:
        raise ValueError("score positions and observed times must have equal length >= 2")
    score_intervals = tuple(right - left for left, right in pairwise(positions))
    time_intervals = tuple(right - left for left, right in pairwise(times))
    if any(value <= 0 for value in score_intervals):
        raise ValueError("score positions must be strictly increasing")
    if any(value < 0 for value in time_intervals):
        raise ValueError("observed times must be non-decreasing")
    slopes = tuple(
        Fraction(time_interval, score_interval)
        for score_interval, time_interval in zip(score_intervals, time_intervals, strict=True)
    )
    return MonotonicTimeExplanation(
        score_positions=positions,
        observed_times=times,
        reconstructed_times=times,
        local_score_intervals=score_intervals,
        segment_slopes=slopes,
    )


def monotonic_score_grid_family_member(
    middle_position: int,
) -> MonotonicTimeExplanation:
    """同じ観測時刻を説明する無限個の楽譜格子から一つを作る。"""
    if middle_position < 2:
        raise ValueError("middle score position must be >= 2")
    return monotonic_time_explanation(
        (0, 1, middle_position, middle_position + 1),
        (0, 1_000, 2_000, 4_000),
    )


def assess_sequential_performance_stage() -> dict[str, Any]:
    """楽譜未知で演奏だけを先に分ける直列段階を成立条件で判定する。"""
    return {
        "schema_version": 1,
        "status": "sequential_performance_first_not_supported",
        "score_unknown": {
            "candidate_set": "unbounded",
            "reconstructable": False,
            "component_swappable": False,
            "copies_full_note_sequence": False,
        },
        "evidence": {
            "score_grid_family": "score_positions=(0,1,n,n+1), integer n >= 2",
            "observed_times_us": [0, 1_000, 2_000, 4_000],
            "reason": (
                "a performance time map cannot be reconstructed or applied to another "
                "score until one score grid candidate is selected"
            ),
        },
        "recommendation": "coupled_score_timing_hypothesis",
    }


def _expected_score_attacks(plan: PiecePlan, score: ScoreSpec) -> tuple[ExpectedScoreAttack, ...]:
    leaves, _ = ordered_leaf_schedule(plan, score)
    materials = {material.material_id: material for material in score.materials}
    attacks: list[ExpectedScoreAttack] = []
    for leaf, start, _ in leaves:
        assert leaf.score_material_id is not None
        material = materials[leaf.score_material_id]
        attacks.extend(
            ExpectedScoreAttack(
                event_id=f"{leaf.node_id}:{note.event_id}",
                score_unit=start + note.at_units,
                pitch=note.pitch,
                voice=note.voice,
            )
            for note in material.notes
        )
    return tuple(sorted(attacks, key=lambda item: (item.score_unit, item.pitch, item.event_id)))


def _coordination_status(pitches: tuple[int, ...], onsets: tuple[int, ...]) -> str:
    spread = max(onsets) - min(onsets)
    if spread == 0:
        return "vertical_equivalence"
    ordered = sorted(zip(pitches, onsets, strict=True), key=lambda item: item[0])
    relative = tuple(onset - min(onsets) for _, onset in ordered)
    if relative == tuple(sorted(relative)) and relative[-1] <= 45_000:
        return "rolled_v1_candidate"
    return "local_residual_unmodeled"


def align_known_score_note_events(
    plan: PiecePlan,
    score: ScoreSpec,
    observed_performance: ObservedPerformance,
) -> KnownScoreNoteAlignment:
    """既知楽譜と観測noteを音高別の非交差順で一対一対応させる。"""
    expected = _expected_score_attacks(plan, score)
    expected_by_pitch: dict[int, list[ExpectedScoreAttack]] = defaultdict(list)
    observed_by_pitch: dict[int, list[ObservedNote]] = defaultdict(list)
    for attack in expected:
        expected_by_pitch[attack.pitch].append(attack)
    for note in observed_performance.notes:
        observed_by_pitch[note.pitch].append(note)
    for notes in observed_by_pitch.values():
        notes.sort(key=lambda item: (item.onset_us, item.note_on_event_id))
    all_pitches = set(expected_by_pitch) | set(observed_by_pitch)
    unmatched_expected = sum(
        max(0, len(expected_by_pitch[pitch]) - len(observed_by_pitch[pitch]))
        for pitch in all_pitches
    )
    unmatched_observed = sum(
        max(0, len(observed_by_pitch[pitch]) - len(expected_by_pitch[pitch]))
        for pitch in all_pitches
    )
    assumptions = ("pitch-local non-crossing alignment",)
    if unmatched_expected or unmatched_observed:
        return KnownScoreNoteAlignment(
            status="unable_to_investigate",
            matches=(),
            assumptions=assumptions,
            unmatched_expected_count=unmatched_expected,
            unmatched_observed_count=unmatched_observed,
        )
    matches = [
        AlignedExpectedObservedAttack(expected=attack, observed=note)
        for pitch in sorted(expected_by_pitch)
        for attack, note in zip(expected_by_pitch[pitch], observed_by_pitch[pitch], strict=True)
    ]
    return KnownScoreNoteAlignment(
        status="assessed",
        matches=tuple(
            sorted(
                matches,
                key=lambda item: (
                    item.expected.score_unit,
                    item.expected.pitch,
                    item.expected.event_id,
                ),
            )
        ),
        assumptions=assumptions,
        unmatched_expected_count=0,
        unmatched_observed_count=0,
    )


def align_known_score_timing(
    plan: PiecePlan,
    score: ScoreSpec,
    observed_smf: ObservedSmf,
    observed_performance: ObservedPerformance,
) -> KnownScoreTimingEvidence:
    """既知楽譜の発音を音高別の非交差順で観測note候補へ対応させる。"""
    alignment = align_known_score_note_events(plan, score, observed_performance)
    if alignment.status != "assessed":
        return KnownScoreTimingEvidence(
            status="unable_to_investigate",
            score_known=True,
            note_on_status="unable_to_investigate",
            note_off_status="unable_to_investigate",
            cc64_status="unable_to_investigate",
            attack_groups=(),
            assumptions=alignment.assumptions,
            unmatched_expected_count=alignment.unmatched_expected_count,
            unmatched_observed_count=alignment.unmatched_observed_count,
        )
    aligned_by_unit: dict[int, list[tuple[ExpectedScoreAttack, ObservedNote]]] = defaultdict(list)
    for match in alignment.matches:
        aligned_by_unit[match.expected.score_unit].append((match.expected, match.observed))
    groups: list[TimingAttackGroupEvidence] = []
    for score_unit, aligned in sorted(aligned_by_unit.items()):
        ordered = sorted(aligned, key=lambda item: (item[0].pitch, item[0].event_id))
        onsets = tuple(int(note.onset_us) for _, note in ordered)
        minimum_anchor = min(onsets)
        median_anchor = round(median(onsets))
        relative = tuple(onset - minimum_anchor for onset in onsets)
        groups.append(
            TimingAttackGroupEvidence(
                score_unit=score_unit,
                evidence_event_ids=tuple(note.note_on_event_id for _, note in ordered),
                minimum_anchor_us=minimum_anchor,
                median_anchor_us=median_anchor,
                minimum_anchor_residuals_us=relative,
                median_anchor_residuals_us=tuple(onset - median_anchor for onset in onsets),
                invariant_relative_onsets_us=relative,
                invariant_spread_us=max(onsets) - min(onsets),
                coordination_status=_coordination_status(
                    tuple(attack.pitch for attack, _ in ordered), onsets
                ),
            )
        )
    has_cc64 = any(
        event.message_type == "control_change" and event.field("control") == 64
        for event in observed_smf.events
    )
    note_off_status = (
        "ambiguous_not_attributed"
        if observed_performance.note_matching_status == "ambiguous"
        else "observed_not_attributed"
    )
    status = "ambiguous" if note_off_status == "ambiguous_not_attributed" else "assessed"
    return KnownScoreTimingEvidence(
        status=status,
        score_known=True,
        note_on_status="assessed",
        note_off_status=note_off_status,
        cc64_status="observed_not_attributed" if has_cc64 else "not_present",
        attack_groups=tuple(groups),
        assumptions=(
            "pitch-local non-crossing alignment",
            "minimum and median anchors are alternatives, not observations",
            "45 ms low-to-high spread is the rolled-v1 compatibility rule",
        ),
        unmatched_expected_count=0,
        unmatched_observed_count=0,
    )


def build_timing_control_artifacts() -> dict[str, Any]:
    """版付きの完全同値と楽譜時間の非同定性対照を作る。"""
    plan = PiecePlan(
        plan_id="timing-control-v1",
        title="timing control",
        tonal_center=0,
        mode="major",
        root_node_id="root",
        ending_intent="tonic",
        nodes=(
            PlanNode("root", None, 0, "whole"),
            PlanNode(
                "leaf",
                "root",
                0,
                "statement",
                duration_weight=1,
                score_material_id="material",
            ),
        ),
    )
    score = ScoreSpec(
        score_id="timing-control-score-v1",
        divisions=4,
        materials=(
            ScoreMaterial(
                material_id="material",
                length_units=8,
                notes=(
                    ScoreNote("low-1", 0, 2, 48, "lower"),
                    ScoreNote("high-1", 0, 2, 64, "upper"),
                    ScoreNote("low-2", 2, 2, 52, "lower"),
                    ScoreNote("high-2", 2, 2, 67, "upper"),
                    ScoreNote("ending-low", 4, 4, 48, "lower"),
                    ScoreNote("ending-high", 4, 4, 60, "upper"),
                ),
            ),
        ),
    )
    base = PerformanceSpec(
        performance_id="timing-control-performance-v1",
        target_duration_ms=8_000,
        default_velocity=64,
        timing_budget_id="narrative-v1",
        node_performances=(
            NodePerformance(
                "root",
                timing_profile="neutral",
                timing_amount="subtle",
                coordination_profile="score",
            ),
        ),
    )
    candidates = {
        "score": base,
        "aligned": replace(
            base,
            node_performances=(replace(base.node_performances[0], coordination_profile="aligned"),),
        ),
        "neutral-moderate": replace(
            base,
            node_performances=(replace(base.node_performances[0], timing_amount="moderate"),),
        ),
        "root-savor-v1": replace(
            base,
            node_performances=(
                replace(
                    base.node_performances[0],
                    timing_profile="savor",
                    timing_amount="moderate",
                ),
            ),
        ),
    }
    equivalence = timing_candidate_equivalence(plan, score, candidates)
    observed_times = (0, 1_000, 2_000, 4_000)
    first = monotonic_score_grid_family_member(3)
    second = monotonic_score_grid_family_member(4)
    return {
        "finite_vocabulary_collisions": {
            "schema_version": 1,
            "candidate_space_id": equivalence.candidate_space_id,
            "candidate_space_sha256": equivalence.candidate_space_sha256,
            "status": equivalence.status,
            "surface_candidate_count": equivalence.surface_candidate_count,
            "semantic_candidate_count": equivalence.semantic_candidate_count,
            "semantic_fingerprints": equivalence.semantic_fingerprints,
            "equivalence_classes": equivalence.equivalence_classes,
        },
        "non_identifiability": {
            "schema_version": 1,
            "status": "affirmative_evidence",
            "same_observed_times": first.reconstructed_times == second.reconstructed_times,
            "different_local_score_intervals": first.local_score_intervals
            != second.local_score_intervals,
            "observed_times": observed_times,
            "candidates": [
                {
                    "score_positions": item.score_positions,
                    "local_score_intervals": item.local_score_intervals,
                    "segment_slopes": [
                        [slope.numerator, slope.denominator] for slope in item.segment_slopes
                    ],
                }
                for item in (first, second)
            ],
            "conclusion": "equivalent_candidates",
            "scope": "general monotonic time maps, not only the current finite vocabulary",
            "unbounded_family": {
                "parameter": "integer n >= 2",
                "score_positions": ["0", "1", "n", "n+1"],
            },
        },
    }
