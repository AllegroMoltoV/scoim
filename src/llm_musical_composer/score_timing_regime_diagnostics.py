"""既知leaf内部で、全曲格子と局所リズム比を比較する診断。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

from llm_musical_composer.performance_pipeline import ordered_leaf_schedule
from llm_musical_composer.pipeline_dsl import parse_piece_plan, parse_score_spec
from llm_musical_composer.reference_decomposition import (
    build_observed_performance,
    load_observed_smf,
)
from llm_musical_composer.reference_timing_hypothesis import align_known_score_timing
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file
from llm_musical_composer.score_timing_hypothesis import (
    VOCABULARIES,
    GroupingAttack,
    IntervalVocabularyFit,
    build_grouping_profiles,
    compare_grouping_to_known_score,
    fit_interval_vocabulary_global,
)


@dataclass(frozen=True)
class LeafSpan:
    node_id: str
    material_id: str
    start_position: int
    end_position: int


@dataclass(frozen=True)
class LeafInternalIntervals:
    leaf: LeafSpan
    interval_indices: tuple[int, ...]


@dataclass(frozen=True)
class LeafIntervalPartition:
    leaves: tuple[LeafInternalIntervals, ...]
    cross_boundary_indices: tuple[int, ...]


@dataclass(frozen=True)
class SelectedVocabularyFit:
    vocabulary_id: str
    fit: IntervalVocabularyFit


@dataclass(frozen=True)
class IntervalRatioComparison:
    status: str
    matching_interval_count: int
    interval_count: int
    inferred_pattern: tuple[int, ...]
    expected_pattern: tuple[int, ...]


@dataclass(frozen=True)
class UniqueGroupingProfile:
    groups: tuple[GroupingAttack, ...]
    aliases: tuple[str, ...]


def partition_leaf_internal_intervals(
    *,
    positions: tuple[int, ...],
    leaves: tuple[LeafSpan, ...],
) -> LeafIntervalPartition:
    """隣接位置をleaf内部または境界横断へ排他的に分類する。"""
    if len(positions) < 2 or any(right <= left for left, right in pairwise(positions)):
        raise ValueError("positions must be strictly increasing and contain two values")
    ordered_leaves = tuple(sorted(leaves, key=lambda leaf: leaf.start_position))
    if any(leaf.end_position <= leaf.start_position for leaf in ordered_leaves):
        raise ValueError("leaf spans must have positive length")
    indices_by_node = {leaf.node_id: [] for leaf in ordered_leaves}
    cross_boundary: list[int] = []
    for index, (left, right) in enumerate(pairwise(positions)):
        owner = next(
            (
                leaf
                for leaf in ordered_leaves
                if leaf.start_position <= left and right < leaf.end_position
            ),
            None,
        )
        if owner is None:
            cross_boundary.append(index)
        else:
            indices_by_node[owner.node_id].append(index)
    return LeafIntervalPartition(
        leaves=tuple(
            LeafInternalIntervals(leaf, tuple(indices_by_node[leaf.node_id]))
            for leaf in ordered_leaves
        ),
        cross_boundary_indices=tuple(cross_boundary),
    )


def select_observed_vocabulary(
    observed_intervals: tuple[int, ...],
) -> SelectedVocabularyFit:
    """既知楽譜を使わず、観測誤差と決定的同率規則で語彙を選ぶ。"""
    fits = tuple(
        SelectedVocabularyFit(
            vocabulary_id,
            fit_interval_vocabulary_global(observed_intervals, vocabulary),
        )
        for vocabulary_id, vocabulary in VOCABULARIES.items()
    )
    return min(
        fits,
        key=lambda selected: (
            selected.fit.squared_error,
            selected.fit.normalized_intervals,
            selected.vocabulary_id,
        ),
    )


def _normalized_pattern(values: tuple[Fraction | int, ...]) -> tuple[int, ...]:
    if not values or any(value <= 0 for value in values):
        raise ValueError("interval values must be positive and non-empty")
    fractions = tuple(Fraction(value) for value in values)
    denominator = math.lcm(*(value.denominator for value in fractions))
    integers = tuple(
        value.numerator * denominator // value.denominator for value in fractions
    )
    divisor = math.gcd(*integers)
    return tuple(value // divisor for value in integers)


def compare_interval_ratios(
    inferred: tuple[Fraction | int, ...],
    expected: tuple[Fraction | int, ...],
) -> IntervalRatioComparison:
    """二列をそれぞれ倍率正規化し、leaf内部比だけを比較する。"""
    if len(inferred) != len(expected):
        raise ValueError("inferred and expected intervals must have equal length")
    inferred_pattern = _normalized_pattern(inferred)
    expected_pattern = _normalized_pattern(expected)
    matching = sum(
        left == right
        for left, right in zip(inferred_pattern, expected_pattern, strict=True)
    )
    return IntervalRatioComparison(
        status="equivalent" if inferred_pattern == expected_pattern else "different",
        matching_interval_count=matching,
        interval_count=len(expected_pattern),
        inferred_pattern=inferred_pattern,
        expected_pattern=expected_pattern,
    )


def deduplicate_grouping_profiles(
    profiles: dict[str, tuple[GroupingAttack, ...]],
) -> dict[str, UniqueGroupingProfile]:
    """完全に同じ発音群列へ、最初のprofile IDと別名をまとめる。"""
    canonical_by_groups: dict[tuple[GroupingAttack, ...], str] = {}
    aliases: dict[str, list[str]] = {}
    groups_by_id: dict[str, tuple[GroupingAttack, ...]] = {}
    for profile_id, groups in profiles.items():
        canonical = canonical_by_groups.setdefault(groups, profile_id)
        aliases.setdefault(canonical, []).append(profile_id)
        groups_by_id.setdefault(canonical, groups)
    return {
        profile_id: UniqueGroupingProfile(
            groups=groups_by_id[profile_id],
            aliases=tuple(aliases[profile_id]),
        )
        for profile_id in aliases
    }


def _fit_record(selected: SelectedVocabularyFit) -> dict[str, Any]:
    return {
        "vocabulary_id": selected.vocabulary_id,
        "scale": [selected.fit.scale.numerator, selected.fit.scale.denominator],
        "squared_error": [
            selected.fit.squared_error.numerator,
            selected.fit.squared_error.denominator,
        ],
        "normalized_intervals": list(selected.fit.normalized_intervals),
    }


def assess_local_rhythm_profile(
    *,
    positions: tuple[int, ...],
    times_us: tuple[int, ...],
    leaves: tuple[LeafSpan, ...],
) -> dict[str, Any]:
    """一つの発音群列について、leaf局所比と全曲格子対照を比較する。"""
    if len(positions) != len(times_us):
        raise ValueError("positions and times must have equal length")
    observed_intervals = tuple(right - left for left, right in pairwise(times_us))
    if any(value <= 0 for value in observed_intervals):
        raise ValueError("observed times must be strictly increasing")
    partition = partition_leaf_internal_intervals(positions=positions, leaves=leaves)
    global_fit = select_observed_vocabulary(observed_intervals)
    leaf_records = []
    local_exact = 0
    global_exact = 0
    local_matching = 0
    global_matching = 0
    evaluated_intervals = 0
    improved = 0
    unchanged = 0
    worsened = 0
    outside = 0
    for item in partition.leaves:
        indices = item.interval_indices
        if not indices:
            outside += 1
            leaf_records.append(
                {
                    "node_id": item.leaf.node_id,
                    "material_id": item.leaf.material_id,
                    "status": "outside_candidate_space",
                    "interval_count": 0,
                }
            )
            continue
        observed = tuple(observed_intervals[index] for index in indices)
        expected = tuple(positions[index + 1] - positions[index] for index in indices)
        local_fit = select_observed_vocabulary(observed)
        local_comparison = compare_interval_ratios(local_fit.fit.intervals, expected)
        global_comparison = compare_interval_ratios(
            tuple(global_fit.fit.intervals[index] for index in indices),
            expected,
        )
        local_exact += local_comparison.status == "equivalent"
        global_exact += global_comparison.status == "equivalent"
        local_matching += local_comparison.matching_interval_count
        global_matching += global_comparison.matching_interval_count
        evaluated_intervals += local_comparison.interval_count
        difference = (
            local_comparison.matching_interval_count
            - global_comparison.matching_interval_count
        )
        if difference > 0:
            improved += 1
            change = "improved"
        elif difference < 0:
            worsened += 1
            change = "worsened"
        else:
            unchanged += 1
            change = "unchanged"
        leaf_records.append(
            {
                "node_id": item.leaf.node_id,
                "material_id": item.leaf.material_id,
                "status": "assessed",
                "interval_count": len(indices),
                "interval_indices": list(indices),
                "local_fit": _fit_record(local_fit),
                "global_fit_slice": {
                    "vocabulary_id": global_fit.vocabulary_id,
                    "normalized_intervals": list(global_comparison.inferred_pattern),
                },
                "expected_pattern": list(local_comparison.expected_pattern),
                "local_matching_interval_count": (
                    local_comparison.matching_interval_count
                ),
                "global_matching_interval_count": (
                    global_comparison.matching_interval_count
                ),
                "local_status": local_comparison.status,
                "global_status": global_comparison.status,
                "change": change,
            }
        )
    return {
        "status": "assessed",
        "group_count": len(positions),
        "evaluated_leaf_count": len(partition.leaves) - outside,
        "outside_candidate_space_leaf_count": outside,
        "evaluated_internal_interval_count": evaluated_intervals,
        "cross_boundary_count": len(partition.cross_boundary_indices),
        "cross_boundary_indices": list(partition.cross_boundary_indices),
        "local_exact_leaf_count": local_exact,
        "global_exact_leaf_count": global_exact,
        "local_matching_interval_count": local_matching,
        "global_matching_interval_count": global_matching,
        "improved_leaf_count": improved,
        "unchanged_leaf_count": unchanged,
        "worsened_leaf_count": worsened,
        "whole_song_fit": _fit_record(global_fit),
        "leaves": leaf_records,
    }


def _source_record(source: object) -> dict[str, Any]:
    required = (source.piece_path, source.score_path, source.smf_path)
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        return {
            "case_id": source.case_id,
            "group_id": source.group_id,
            "status": "unable_to_investigate",
            "missing": missing,
        }
    plan = parse_piece_plan(source.piece_path.read_text(encoding="utf-8"))
    score = parse_score_spec(source.score_path.read_text(encoding="utf-8"))
    observed_smf = load_observed_smf(source.smf_path)
    observed_performance = build_observed_performance(observed_smf)
    timing_evidence = align_known_score_timing(
        plan,
        score,
        observed_smf,
        observed_performance,
    )
    if timing_evidence.note_on_status != "assessed":
        return {
            "case_id": source.case_id,
            "group_id": source.group_id,
            "status": timing_evidence.status,
            "reason": "known score timing could not be aligned",
            "unmatched_expected_count": timing_evidence.unmatched_expected_count,
            "unmatched_observed_count": timing_evidence.unmatched_observed_count,
        }
    known_units = {
        event_id: group.score_unit
        for group in timing_evidence.attack_groups
        for event_id in group.evidence_event_ids
    }
    schedule, _ = ordered_leaf_schedule(plan, score)
    leaves = tuple(
        LeafSpan(
            node_id=leaf.node_id,
            material_id=leaf.score_material_id or "",
            start_position=start,
            end_position=end,
        )
        for leaf, start, end in schedule
    )
    unique_profiles = deduplicate_grouping_profiles(
        build_grouping_profiles(observed_performance)
    )
    profile_records = []
    for profile_id, profile in unique_profiles.items():
        grouping = compare_grouping_to_known_score(profile.groups, known_units)
        if grouping["status"] != "compatible":
            profile_records.append(
                {
                    "profile_id": profile_id,
                    "aliases": list(profile.aliases),
                    "status": "grouping_incompatible",
                    "grouping_diagnostics": grouping,
                }
            )
            continue
        positions = tuple(
            known_units[group.source_event_ids[0]] for group in profile.groups
        )
        times_us = tuple(group.onset_us for group in profile.groups)
        assessment = assess_local_rhythm_profile(
            positions=positions,
            times_us=times_us,
            leaves=leaves,
        )
        profile_records.append(
            {
                "profile_id": profile_id,
                "aliases": list(profile.aliases),
                "status": "assessed",
                "grouping_diagnostics": grouping,
                "assessment": assessment,
            }
        )
    assessed = [record for record in profile_records if record["status"] == "assessed"]
    return {
        "case_id": source.case_id,
        "group_id": source.group_id,
        "status": "assessed" if assessed else "unable_to_investigate",
        "profile_count_before_deduplication": 3,
        "unique_profile_count": len(unique_profiles),
        "assessed_profile_count": len(assessed),
        "profiles": profile_records,
    }


def run_known_local_rhythm_regime_diagnostics(
    *, sources: tuple[object, ...], output_dir: Path
) -> dict[str, Any]:
    """既知生成元すべてへleaf内部比の局所・全曲対照を適用する。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = tuple(_source_record(source) for source in sources)
    jsonl_path = output_dir / "known-source-local-rhythm.jsonl"
    atomic_write_bytes(
        jsonl_path,
        b"".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
            for record in records
        ),
    )
    assessed_records = tuple(record for record in records if record["status"] == "assessed")
    assessments = tuple(
        profile["assessment"]
        for record in assessed_records
        for profile in record["profiles"]
        if profile["status"] == "assessed"
    )
    result = {
        "status": "pass" if len(assessed_records) == len(records) else "partial",
        "source_count": len(records),
        "assessed_source_count": len(assessed_records),
        "assessed_profile_count": len(assessments),
        "local_exact_leaf_count": sum(
            item["local_exact_leaf_count"] for item in assessments
        ),
        "global_exact_leaf_count": sum(
            item["global_exact_leaf_count"] for item in assessments
        ),
        "local_matching_interval_count": sum(
            item["local_matching_interval_count"] for item in assessments
        ),
        "global_matching_interval_count": sum(
            item["global_matching_interval_count"] for item in assessments
        ),
        "evaluated_internal_interval_count": sum(
            item["evaluated_internal_interval_count"] for item in assessments
        ),
        "cross_boundary_count": sum(item["cross_boundary_count"] for item in assessments),
    }
    atomic_write_json(output_dir / "result.json", result)
    manifest = {
        "schema_version": 1,
        "status": result["status"],
        "outputs": {
            "known-source-local-rhythm.jsonl": sha256_file(jsonl_path),
            "result.json": sha256_file(output_dir / "result.json"),
        },
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return result


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    from llm_musical_composer.reference_timing_run import default_known_timing_sources

    arguments = _parse_arguments()
    result = run_known_local_rhythm_regime_diagnostics(
        sources=default_known_timing_sources(arguments.workspace),
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
