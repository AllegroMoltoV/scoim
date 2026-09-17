"""ScoreTiming候補を現行PerformanceSpecへ早期確定できる範囲を診断する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise, product
from pathlib import Path
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
    performance_time_map,
    render_performance,
)
from llm_musical_composer.pipeline_dsl import (
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.reference_decomposition import (
    build_observed_performance,
    load_observed_smf,
)
from llm_musical_composer.reference_timing_hypothesis import align_known_score_timing
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file
from llm_musical_composer.score_timing_hypothesis import (
    GroupingAttack,
    ScoreTimingHypothesisV0,
    build_grouping_profiles,
    compare_candidate_to_known_score,
    generate_score_timing_candidates,
)
from llm_musical_composer.score_timing_known_fixtures import (
    KnownTimingFixture,
    build_known_timing_fixtures,
)

_BUDGETS = ("subtle-v1", "narrative-v1", "narrative-v2")
_PROFILES = ("neutral", "savor", "flow", "build", "release")
_AMOUNTS = ("subtle", "moderate")


@dataclass(frozen=True)
class TimingDistance:
    status: str
    mean_absolute_error_us: float
    maximum_absolute_error_us: int
    over_one_transport_tick_count: int
    fitted_times_us: tuple[int, ...]


@dataclass(frozen=True)
class FlatTimingSurfaceEntry:
    candidate_id: str
    performance: PerformanceSpec
    time_map_ms: tuple[int, ...]


@dataclass(frozen=True)
class FlatTimingSearchSpace:
    scope: str
    surface_entries: tuple[FlatTimingSurfaceEntry, ...]
    semantic_maps: tuple[tuple[int, ...], ...]


def _stable_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def build_flat_timing_fixture(
    score_positions: tuple[int, ...],
    *,
    material_length_units: int | None = None,
) -> tuple[PiecePlan, ScoreSpec]:
    """候補位置だけを持つ、root兼leafの診断用IRを作る。"""
    positions = tuple(int(value) for value in score_positions)
    if len(positions) < 2 or positions[0] != 0:
        raise ValueError("score positions must start at zero and contain at least two values")
    if any(right <= left for left, right in pairwise(positions)):
        raise ValueError("score positions must be strictly increasing")
    minimum_length = positions[-1] + 1
    length = minimum_length if material_length_units is None else int(material_length_units)
    if length < minimum_length:
        raise ValueError("material length must extend beyond the last attack")
    node = PlanNode(
        node_id="flat-root-leaf",
        parent_id=None,
        order=0,
        role="whole",
        duration_weight=1,
        score_material_id="flat-material",
    )
    plan = PiecePlan(
        plan_id="flat-timing-plan-v1",
        title="flat timing diagnostic",
        tonal_center=0,
        mode="major",
        root_node_id=node.node_id,
        ending_intent="tonic",
        nodes=(node,),
    )
    notes = tuple(
        ScoreNote(
            event_id=f"flat-note-{index}",
            at_units=position,
            duration_units=1,
            pitch=60 + index % 12,
            voice="upper",
        )
        for index, position in enumerate(positions)
    )
    score = ScoreSpec(
        score_id="flat-timing-score-v1",
        divisions=1,
        materials=(
            ScoreMaterial(
                material_id="flat-material",
                length_units=length,
                notes=notes,
            ),
        ),
    )
    return plan, score


def enumerate_flat_timing_vocabulary(
    plan: PiecePlan,
    score: ScoreSpec,
) -> FlatTimingSearchSpace:
    """単一nodeに対する現行30表記を列挙する。"""
    node_id = plan.root_node_id
    total_units = score.materials[0].length_units
    entries: list[FlatTimingSurfaceEntry] = []
    for budget, profile, amount in product(_BUDGETS, _PROFILES, _AMOUNTS):
        candidate_id = f"{budget}--{profile}--{amount}"
        performance = PerformanceSpec(
            performance_id=f"flat-{candidate_id}",
            target_duration_ms=total_units * 1_000,
            default_velocity=64,
            timing_budget_id=budget,
            node_performances=(
                NodePerformance(
                    node_id=node_id,
                    timing_profile=profile,
                    timing_amount=amount,
                    dynamics_profile="steady",
                    articulation_profile="score",
                    coordination_profile="score",
                    pedal_profile="none",
                ),
            ),
        )
        entries.append(
            FlatTimingSurfaceEntry(
                candidate_id=candidate_id,
                performance=performance,
                time_map_ms=performance_time_map(plan, score, performance),
            )
        )
    semantic_maps = tuple(sorted({entry.time_map_ms for entry in entries}))
    return FlatTimingSearchSpace(
        scope="flat_single_node_minimum_tail",
        surface_entries=tuple(entries),
        semantic_maps=semantic_maps,
    )


def compare_timing_shapes(
    reference_times_us: tuple[int, ...],
    candidate_times: tuple[int, ...],
) -> TimingDistance:
    """両端を合わせ、内部時刻の距離をmicrosecond単位で返す。"""
    reference = tuple(int(value) for value in reference_times_us)
    candidate = tuple(int(value) for value in candidate_times)
    if len(reference) != len(candidate) or len(reference) < 2:
        raise ValueError("timing sequences must have equal length >= 2")
    reference_span = reference[-1] - reference[0]
    candidate_span = candidate[-1] - candidate[0]
    if reference_span <= 0 or candidate_span <= 0:
        raise ValueError("timing sequence endpoints must be strictly increasing")
    fitted = tuple(
        reference[0] + round(Fraction((value - candidate[0]) * reference_span, candidate_span))
        for value in candidate
    )
    errors = tuple(
        abs(expected - actual) for expected, actual in zip(reference, fitted, strict=True)
    )
    maximum = max(errors)
    return TimingDistance(
        status=(
            "within_one_transport_tick" if maximum <= 1_000 else "more_than_one_transport_tick"
        ),
        mean_absolute_error_us=round(statistics.fmean(errors), 3),
        maximum_absolute_error_us=maximum,
        over_one_transport_tick_count=sum(error > 1_000 for error in errors),
        fitted_times_us=fitted,
    )


def _distance_record(distance: TimingDistance) -> dict[str, Any]:
    return {
        "status": distance.status,
        "mean_absolute_error_us": distance.mean_absolute_error_us,
        "maximum_absolute_error_us": distance.maximum_absolute_error_us,
        "over_one_transport_tick_count": distance.over_one_transport_tick_count,
    }


def _two_node_fixture(score_positions: tuple[int, ...]) -> tuple[PiecePlan, ScoreSpec]:
    _, score = build_flat_timing_fixture(score_positions)
    root = PlanNode("two-root", None, 0, "whole")
    leaf = PlanNode(
        "two-leaf",
        root.node_id,
        0,
        "statement",
        duration_weight=1,
        score_material_id=score.materials[0].material_id,
    )
    return (
        PiecePlan(
            plan_id="two-node-timing-plan-v1",
            title="two node timing diagnostic",
            tonal_center=0,
            mode="major",
            root_node_id=root.node_id,
            ending_intent="tonic",
            nodes=(root, leaf),
        ),
        score,
    )


def _fixed_performance(
    plan: PiecePlan,
    score: ScoreSpec,
    node_performances: tuple[NodePerformance, ...],
    *,
    performance_id: str,
) -> PerformanceSpec:
    return PerformanceSpec(
        performance_id=performance_id,
        target_duration_ms=score.materials[0].length_units * 1_000,
        default_velocity=64,
        timing_budget_id="subtle-v1",
        node_performances=node_performances,
    )


def build_timing_dependency_counterexamples() -> dict[str, Any]:
    """構成木と末尾長が時間写像へ影響する固定反例を作る。"""
    positions = (0, 25, 50, 75, 100)
    flat_plan, flat_score = build_flat_timing_fixture(positions)
    flat_search = enumerate_flat_timing_vocabulary(flat_plan, flat_score)
    two_plan, two_score = _two_node_fixture(positions)
    composed = _fixed_performance(
        two_plan,
        two_score,
        (
            NodePerformance("two-root", timing_profile="savor", timing_amount="subtle"),
            NodePerformance("two-leaf", timing_profile="neutral", timing_amount="subtle"),
        ),
        performance_id="topology-counterexample",
    )
    composed_map = performance_time_map(two_plan, two_score, composed)
    nearest = min(
        (
            compare_timing_shapes(
                tuple(value * 1_000 for value in composed_map),
                semantic_map,
            )
            for semantic_map in flat_search.semantic_maps
        ),
        key=lambda value: (
            value.maximum_absolute_error_us,
            value.mean_absolute_error_us,
        ),
    )

    dense_positions = tuple(range(101))
    short_plan, short_score = build_flat_timing_fixture(
        dense_positions,
        material_length_units=101,
    )
    long_plan, long_score = build_flat_timing_fixture(
        dense_positions,
        material_length_units=151,
    )
    short_performance = _fixed_performance(
        short_plan,
        short_score,
        (
            NodePerformance(
                short_plan.root_node_id,
                timing_profile="savor",
                timing_amount="moderate",
            ),
        ),
        performance_id="tail-short",
    )
    long_performance = _fixed_performance(
        long_plan,
        long_score,
        (
            NodePerformance(
                long_plan.root_node_id,
                timing_profile="savor",
                timing_amount="moderate",
            ),
        ),
        performance_id="tail-long",
    )
    short_map = performance_time_map(short_plan, short_score, short_performance)
    long_map = performance_time_map(long_plan, long_score, long_performance)
    short_tail_search = enumerate_flat_timing_vocabulary(short_plan, short_score)
    nearest_short_tail_distance, nearest_short_tail_entry = min(
        (
            (
                compare_timing_shapes(
                    tuple(long_map[position] * 1_000 for position in dense_positions),
                    tuple(entry.time_map_ms[position] for position in dense_positions),
                ),
                entry,
            )
            for entry in short_tail_search.surface_entries
        ),
        key=lambda item: (
            item[0].maximum_absolute_error_us,
            item[0].mean_absolute_error_us,
            item[1].candidate_id,
        ),
    )
    tail_distance = compare_timing_shapes(
        tuple(short_map[position] * 1_000 for position in dense_positions),
        tuple(long_map[position] for position in dense_positions),
    )
    short_span_us = (short_map[100] - short_map[0]) * 1_000
    normalized_difference = (
        tail_distance.maximum_absolute_error_us / short_span_us if short_span_us else 0.0
    )
    return {
        "schema_version": 1,
        "topology_dependency": {
            "status": (
                "affirmative_evidence"
                if composed_map not in flat_search.semantic_maps
                else "investigated_no_evidence"
            ),
            "flat_surface_count": len(flat_search.surface_entries),
            "flat_semantic_count": len(flat_search.semantic_maps),
            "composed_map_in_flat_semantic_maps": composed_map in flat_search.semantic_maps,
            "nearest_flat_maximum_error_us": nearest.maximum_absolute_error_us,
            "root_profile": "savor/subtle",
            "leaf_profile": "neutral/subtle",
        },
        "tail_length_dependency": {
            "status": (
                "affirmative_evidence"
                if nearest_short_tail_distance.status == "more_than_one_transport_tick"
                else "investigated_no_evidence"
            ),
            "attack_range_units": [0, 100],
            "material_lengths_units": [101, 151],
            "profile": "savor/moderate",
            "maximum_aligned_error_us": tail_distance.maximum_absolute_error_us,
            "maximum_normalized_difference": round(normalized_difference, 9),
            "short_tail_surface_count": len(short_tail_search.surface_entries),
            "nearest_short_tail_vocabulary_fit": {
                "candidate_id": nearest_short_tail_entry.candidate_id,
                **_distance_record(nearest_short_tail_distance),
            },
        },
    }


def classify_local_coordination(groups: tuple[GroupingAttack, ...]) -> dict[str, Any]:
    """発音群の局所offsetを現行coordination語彙と照合する。"""
    if all(all(offset == 0 for offset in group.relative_onsets_us) for group in groups):
        return {
            "status": "representable",
            "profile_semantics": ["aligned", "score"],
            "oracle_assumption": None,
        }
    rolled_matches = True
    for group in groups:
        offsets = group.relative_onsets_us
        divisor = max(1, len(offsets) - 1)
        expected = tuple(round(45 * index / divisor) * 1_000 for index in range(len(offsets)))
        if offsets != expected:
            rolled_matches = False
            break
    if rolled_matches:
        return {
            "status": "representable_under_pitch_rank_voice_assumption",
            "profile_semantics": ["rolled"],
            "oracle_assumption": "group members follow the renderer low-to-high ordering",
        }
    return {
        "status": "not_represented",
        "profile_semantics": [],
        "oracle_assumption": None,
    }


def _candidate_attack_times_us(candidate: ScoreTimingHypothesisV0) -> tuple[int, ...]:
    knots = candidate.shared_time_map
    if len(knots) < 2:
        raise ValueError("candidate time map must contain at least two knots")
    times: list[int] = []
    segment_index = 0
    for reference in candidate.attack_group_refs:
        position = reference.score_position
        while (
            segment_index + 1 < len(knots) - 1
            and position > knots[segment_index + 1].score_position
        ):
            segment_index += 1
        left = knots[segment_index]
        right = knots[segment_index + 1]
        if not left.score_position <= position <= right.score_position:
            raise ValueError("attack position is outside the shared time map")
        offset = Fraction(
            (position - left.score_position) * (right.observed_time_us - left.observed_time_us),
            right.score_position - left.score_position,
        )
        times.append(left.observed_time_us + round(offset))
    return tuple(times)


def _flat_fit(candidate: ScoreTimingHypothesisV0) -> tuple[dict[str, Any], dict[str, Any]]:
    positions = tuple(reference.score_position for reference in candidate.attack_group_refs)
    target_times = _candidate_attack_times_us(candidate)
    plan, score = build_flat_timing_fixture(positions)
    search = enumerate_flat_timing_vocabulary(plan, score)
    assessed = []
    for entry in search.surface_entries:
        rendered_times = tuple(entry.time_map_ms[position] for position in positions)
        distance = compare_timing_shapes(target_times, rendered_times)
        assessed.append((distance, entry))
    best_distance, best_entry = min(
        assessed,
        key=lambda item: (
            item[0].maximum_absolute_error_us,
            item[0].mean_absolute_error_us,
            item[1].candidate_id,
        ),
    )
    fit = {
        "scope": search.scope,
        "tail_assumption": "minimum_positive_tail=1 unit",
        "surface_candidate_count": len(search.surface_entries),
        "semantic_candidate_count": len(search.semantic_maps),
        "best_candidate_id": best_entry.candidate_id,
        **_distance_record(best_distance),
    }
    search_record = {
        "score_position_sha256": _stable_hash(positions),
        "surface_candidate_count": len(search.surface_entries),
        "semantic_candidate_count": len(search.semantic_maps),
        "scope": search.scope,
    }
    return fit, search_record


def _known_units(timing_evidence: object) -> dict[str, int]:
    return {
        event_id: group.score_unit
        for group in timing_evidence.attack_groups
        for event_id in group.evidence_event_ids
    }


def _characterize_original_timing(
    fixture: KnownTimingFixture,
    timing_evidence: object,
) -> dict[str, Any]:
    true_map = performance_time_map(fixture.plan, fixture.score, fixture.performance)
    expected = tuple(group.minimum_anchor_us for group in timing_evidence.attack_groups)
    actual = tuple(true_map[group.score_unit] for group in timing_evidence.attack_groups)
    distance = compare_timing_shapes(expected, actual)
    return {
        "status": "pass" if distance.maximum_absolute_error_us <= 1_000 else "fail",
        **_distance_record(distance),
    }


def _node_boundary_characterization(
    fixture: KnownTimingFixture,
    candidate: ScoreTimingHypothesisV0,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    _, intervals = ordered_leaf_schedule(fixture.plan, fixture.score)
    scale_raw = comparison.get("best_scale")
    if not isinstance(scale_raw, list) or len(scale_raw) != 2:
        return {"status": "unavailable", "reason": "score scale is unavailable"}
    scale = Fraction(scale_raw[0], scale_raw[1])
    candidate_min = candidate.attack_group_refs[0].score_position
    candidate_max = candidate.attack_group_refs[-1].score_position
    known_min = min(group[0] for group in intervals.values())
    root_end = intervals[fixture.plan.root_node_id][1]
    records = []
    for node_id, (start, end) in sorted(intervals.items()):
        for side, unit in (("start", start), ("end", end)):
            inferred = Fraction(candidate_min) + (unit - known_min) * scale
            if inferred.denominator != 1 or not candidate_min <= inferred <= candidate_max:
                status = "unavailable_outside_candidate_domain"
                time_us = None
            else:
                status = "available_characterization"
                time_us = _interpolate_candidate_position(candidate, int(inferred))
            records.append(
                {
                    "node_id": node_id,
                    "side": side,
                    "known_score_unit": unit,
                    "status": status,
                    "candidate_time_us": time_us,
                }
            )
    root_end_record = next(
        item
        for item in records
        if item["node_id"] == fixture.plan.root_node_id
        and item["side"] == "end"
        and item["known_score_unit"] == root_end
    )
    return {
        "status": "characterization_only",
        "available_boundary_count": sum(
            item["status"] == "available_characterization" for item in records
        ),
        "unavailable_boundary_count": sum(
            item["status"] != "available_characterization" for item in records
        ),
        "root_end_status": root_end_record["status"],
        "boundaries": records,
    }


def _interpolate_candidate_position(candidate: ScoreTimingHypothesisV0, position: int) -> int:
    knots = candidate.shared_time_map
    for left, right in pairwise(knots):
        if left.score_position <= position <= right.score_position:
            return left.observed_time_us + round(
                Fraction(
                    (position - left.score_position)
                    * (right.observed_time_us - left.observed_time_us),
                    right.score_position - left.score_position,
                )
            )
    raise ValueError("position is outside candidate time map")


def _verify_fixture_source(fixture: KnownTimingFixture, record: dict[str, Any]) -> None:
    paths = {
        "piece": fixture.source.piece_path,
        "score": fixture.source.score_path,
        "performance": fixture.source.performance_path,
        "smf": fixture.source.smf_path,
    }
    hashes = record.get("hashes")
    if not isinstance(hashes, dict):
        raise ValueError(f"fixture hashes are missing: {fixture.fixture_id}")
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != hashes.get(name):
            raise ValueError(f"fixture hash mismatch: {fixture.fixture_id}:{name}")
    if parse_piece_plan(fixture.source.piece_path.read_text(encoding="utf-8")) != fixture.plan:
        raise ValueError(f"fixture PiecePlan mismatch: {fixture.fixture_id}")
    if parse_score_spec(fixture.source.score_path.read_text(encoding="utf-8")) != fixture.score:
        raise ValueError(f"fixture ScoreSpec mismatch: {fixture.fixture_id}")
    if (
        parse_performance_spec(fixture.source.performance_path.read_text(encoding="utf-8"))
        != fixture.performance
    ):
        raise ValueError(f"fixture PerformanceSpec mismatch: {fixture.fixture_id}")
    evidence = json.loads(fixture.source.evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "passed" or evidence.get("lineage") != list(
        render_performance(fixture.plan, fixture.score, fixture.performance).lineage
    ):
        raise ValueError(f"fixture lineage mismatch: {fixture.fixture_id}")


def _assess_fixture(
    fixture: KnownTimingFixture,
    fixture_record: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _verify_fixture_source(fixture, fixture_record)
    observed_smf = load_observed_smf(fixture.source.smf_path)
    observed = build_observed_performance(observed_smf)
    timing_evidence = align_known_score_timing(
        fixture.plan,
        fixture.score,
        observed_smf,
        observed,
    )
    if timing_evidence.note_on_status != "assessed":
        return (
            [],
            [],
            {
                "fixture_id": fixture.fixture_id,
                "fixture_family_id": fixture.fixture_id,
                "status": "unable_to_investigate",
                "reason": "known note-on alignment is unavailable",
            },
        )
    original = _characterize_original_timing(fixture, timing_evidence)
    if original["status"] != "within_one_transport_tick":
        return (
            [],
            [],
            {
                "fixture_id": fixture.fixture_id,
                "fixture_family_id": fixture.fixture_id,
                "status": "unable_to_investigate",
                "reason": "original PerformanceSpec characterization failed",
                "original_timing": original,
            },
        )
    known_units = _known_units(timing_evidence)
    grouping_profiles = build_grouping_profiles(observed)
    candidate_set = generate_score_timing_candidates(
        observed,
        source_ledger_sha256=observed_smf.ledger_sha256,
    )
    rows: list[dict[str, Any]] = []
    searches: dict[str, dict[str, Any]] = {}
    equivalent_count = 0
    upstream_mismatch_count = 0
    grouping_conflict_count = 0
    within_tick_count = 0
    for candidate in candidate_set.candidates:
        groups = grouping_profiles[candidate.grouping_profile_id]
        comparison = compare_candidate_to_known_score(candidate, groups, known_units)
        comparison_status = comparison.get("status")
        row: dict[str, Any] = {
            "fixture_id": fixture.fixture_id,
            "fixture_family_id": fixture.fixture_id,
            "expected_score_grid_membership": fixture.expected_score_grid_membership,
            "candidate_id": candidate.candidate_id,
            "candidate_search_status": candidate.search_status,
            "grouping_profile_id": candidate.grouping_profile_id,
            "score_grid_id": candidate.score_grid_id,
            "requested_segment_count": candidate.requested_segment_count,
            "known_score_comparison_status": comparison_status,
            "local_coordination": classify_local_coordination(groups),
        }
        if comparison_status == "equivalent":
            equivalent_count += 1
            fit, search = _flat_fit(candidate)
            row["status"] = "flat_single_node_assessed"
            row["flat_fit"] = fit
            row["known_node_boundaries"] = _node_boundary_characterization(
                fixture,
                candidate,
                comparison,
            )
            searches.setdefault(search["score_position_sha256"], search)
            within_tick_count += fit["status"] == "within_one_transport_tick"
        elif comparison_status == "different":
            upstream_mismatch_count += 1
            row["status"] = "upstream_score_grid_mismatch"
            row["flat_fit"] = None
            row["known_node_boundaries"] = None
        else:
            grouping_conflict_count += 1
            row["status"] = "grouping_conflict"
            row["flat_fit"] = None
            row["known_node_boundaries"] = None
        rows.append(row)
    source = {
        "fixture_id": fixture.fixture_id,
        "fixture_family_id": fixture.fixture_id,
        "status": "assessed",
        "candidate_set_status": candidate_set.status,
        "candidate_count": len(candidate_set.candidates),
        "equivalent_score_grid_candidate_count": equivalent_count,
        "upstream_score_grid_mismatch_candidate_count": upstream_mismatch_count,
        "grouping_conflict_candidate_count": grouping_conflict_count,
        "flat_within_one_transport_tick_candidate_count": within_tick_count,
        "outside_score_timing_candidate_space": (
            fixture.expected_score_grid_membership == "outside"
        ),
        "original_timing": original,
    }
    search_rows = [
        {"fixture_id": fixture.fixture_id, **record} for _, record in sorted(searches.items())
    ]
    return rows, search_rows, source


def run_timing_vocabulary_fit(*, workspace: Path, output_dir: Path) -> dict[str, Any]:
    """4人工fixtureを同一条件で診断し、決定的成果物を書く。"""
    workspace = Path(workspace)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    fixture_root = workspace / ".appendix/score-timing-known-fixtures-v1"
    root_result = json.loads((fixture_root / "result.json").read_text(encoding="utf-8"))
    root_manifest = json.loads((fixture_root / "manifest.json").read_text(encoding="utf-8"))
    if root_result.get("status") != "passed" or root_manifest.get("status") != "passed":
        raise ValueError("known fixture root is not passed")
    fixture_records = {
        record["fixture_id"]: record
        for record in (
            json.loads(line)
            for line in (fixture_root / "fixtures.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    fixtures = build_known_timing_fixtures(fixture_root)
    candidate_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        rows, searches, source = _assess_fixture(fixture, fixture_records[fixture.fixture_id])
        candidate_rows.extend(rows)
        search_rows.extend(searches)
        source_rows.append(source)
    counterexamples = build_timing_dependency_counterexamples()
    assessed_sources = [row for row in source_rows if row["status"] == "assessed"]
    upstream_sources = [
        row for row in assessed_sources if row["upstream_score_grid_mismatch_candidate_count"] > 0
    ]
    recovered_sources = [
        row for row in assessed_sources if row["equivalent_score_grid_candidate_count"] > 0
    ]
    flat_assessed_count = sum(
        row["status"] == "flat_single_node_assessed" for row in candidate_rows
    )
    upstream_mismatch_count = sum(
        row["status"] == "upstream_score_grid_mismatch" for row in candidate_rows
    )
    grouping_conflict_count = sum(row["status"] == "grouping_conflict" for row in candidate_rows)
    local_status_counts = {
        status: sum(row["local_coordination"]["status"] == status for row in candidate_rows)
        for status in (
            "representable",
            "representable_under_pitch_rank_voice_assumption",
            "not_represented",
        )
    }
    root_end_unavailable_count = sum(
        isinstance(row.get("known_node_boundaries"), dict)
        and row["known_node_boundaries"].get("root_end_status")
        == "unavailable_outside_candidate_domain"
        for row in candidate_rows
    )
    evidence_complete = (
        len(fixtures) == 4
        and len(assessed_sources) == len(fixtures)
        and all(row["candidate_count"] == 27 for row in assessed_sources)
        and len(candidate_rows) == 108
        and len(candidate_rows)
        == flat_assessed_count + upstream_mismatch_count + grouping_conflict_count
        and all(count > 0 for count in local_status_counts.values())
        and root_end_unavailable_count == flat_assessed_count
        and counterexamples["topology_dependency"]["status"] == "affirmative_evidence"
        and counterexamples["tail_length_dependency"]["status"] == "affirmative_evidence"
    )
    result = {
        "schema_version": 1,
        "status": "pass" if evidence_complete else "partial",
        "fixture_source_count": len(source_rows),
        "fixture_family_count": len({row["fixture_family_id"] for row in source_rows}),
        "assessed_source_count": len(assessed_sources),
        "candidate_count": len(candidate_rows),
        "flat_assessed_candidate_count": flat_assessed_count,
        "upstream_score_grid_mismatch_candidate_count": upstream_mismatch_count,
        "grouping_conflict_candidate_count": grouping_conflict_count,
        "flat_within_one_transport_tick_candidate_count": sum(
            row.get("flat_fit", {}).get("status") == "within_one_transport_tick"
            for row in candidate_rows
            if isinstance(row.get("flat_fit"), dict)
        ),
        "upstream_score_grid_mismatch_source_count": len(upstream_sources),
        "score_grid_recovered_source_count": len(recovered_sources),
        "score_grid_unrecovered_source_count": len(assessed_sources) - len(recovered_sources),
        "aligned_local_coordination_candidate_count": local_status_counts["representable"],
        "rolled_local_coordination_candidate_count": local_status_counts[
            "representable_under_pitch_rank_voice_assumption"
        ],
        "unrepresented_local_coordination_candidate_count": local_status_counts["not_represented"],
        "root_end_unavailable_candidate_count": root_end_unavailable_count,
        "outside_candidate_space_source_count": sum(
            row.get("outside_score_timing_candidate_space") is True for row in source_rows
        ),
        "topology_dependency_status": counterexamples["topology_dependency"]["status"],
        "tail_length_dependency_status": counterexamples["tail_length_dependency"]["status"],
        "full_performance_mapping_status": "deferred_until_form_and_tail_duration",
        "scope": (
            "shared note-on shape in a flat single-node minimum-tail diagnostic; "
            "not full PerformanceSpec representability"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(output_dir / "candidate-results.jsonl", _jsonl_bytes(candidate_rows))
    atomic_write_json(output_dir / "counterexamples.json", counterexamples)
    atomic_write_bytes(output_dir / "semantic-search-spaces.jsonl", _jsonl_bytes(search_rows))
    atomic_write_json(output_dir / "source-results.json", {"sources": source_rows})
    atomic_write_json(output_dir / "experiment-result.json", result)
    output_names = (
        "candidate-results.jsonl",
        "counterexamples.json",
        "experiment-result.json",
        "semantic-search-spaces.jsonl",
        "source-results.json",
    )
    input_paths = {
        "fixtures.jsonl": fixture_root / "fixtures.jsonl",
        "fixtures-manifest.json": fixture_root / "manifest.json",
        "pipeline_dsl.py": workspace / "src/llm_musical_composer/pipeline_dsl.py",
        "performance_pipeline.py": workspace / "src/llm_musical_composer/performance_pipeline.py",
        "pyproject.toml": workspace / "pyproject.toml",
        "reference_decomposition.py": workspace
        / "src/llm_musical_composer/reference_decomposition.py",
        "reference_timing_hypothesis.py": workspace
        / "src/llm_musical_composer/reference_timing_hypothesis.py",
        "run_state.py": workspace / "src/llm_musical_composer/run_state.py",
        "score_timing_hypothesis.py": workspace
        / "src/llm_musical_composer/score_timing_hypothesis.py",
        "score_timing_known_fixtures.py": workspace
        / "src/llm_musical_composer/score_timing_known_fixtures.py",
        "timing_vocabulary_fit.py": workspace / "src/llm_musical_composer/timing_vocabulary_fit.py",
    }
    for fixture in fixtures:
        prefix = fixture.fixture_id
        input_paths[f"{prefix}/piece-plan.dsl"] = fixture.source.piece_path
        input_paths[f"{prefix}/score-spec.dsl"] = fixture.source.score_path
        input_paths[f"{prefix}/performance-spec.dsl"] = fixture.source.performance_path
        input_paths[f"{prefix}/final.mid"] = fixture.source.smf_path
        input_paths[f"{prefix}/result.json"] = fixture.source.evidence_path
    manifest = {
        "schema_version": 1,
        "status": result["status"],
        "inputs": {name: sha256_file(path) for name, path in sorted(input_paths.items())},
        "outputs": {name: sha256_file(output_dir / name) for name in output_names},
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return result


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    result = run_timing_vocabulary_fit(
        workspace=arguments.workspace,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
