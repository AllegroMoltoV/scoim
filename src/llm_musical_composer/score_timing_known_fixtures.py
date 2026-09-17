"""異なる設計族のScoreTiming既知生成元を決定的に作る。"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, pairwise
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
    render_performance_smf,
    resolve_effective_profile,
    validate_pipeline,
)
from llm_musical_composer.pipeline_dsl import (
    dump_performance_spec,
    dump_piece_plan,
    dump_score_spec,
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.reference_decomposition import (
    build_observed_performance,
    load_observed_smf,
)
from llm_musical_composer.reference_timing_hypothesis import align_known_score_timing
from llm_musical_composer.reference_timing_run import KnownTimingSource
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file
from llm_musical_composer.score_timing_hypothesis import (
    VOCABULARIES,
    build_grouping_profiles,
    compare_candidate_to_known_score,
    compare_grouping_to_known_score,
    generate_score_timing_candidates,
)
from llm_musical_composer.score_timing_oracle_diagnostics import (
    fit_equal_piecewise_time_map,
)


@dataclass(frozen=True)
class KnownTimingFixture:
    fixture_id: str
    expected_score_grid_membership: str
    expected_grouping_profiles: tuple[str, ...]
    plan: PiecePlan
    score: ScoreSpec
    performance: PerformanceSpec
    source: KnownTimingSource


@dataclass(frozen=True)
class _FixtureRecipe:
    fixture_id: str
    prefix: str
    intervals: tuple[int, ...]
    template: tuple[tuple[int, ...], ...]
    permutation: tuple[int, ...]
    expected_score_grid_membership: str
    expected_grouping_profiles: tuple[str, ...]
    profiles: tuple[str, ...]
    amounts: tuple[str, ...]
    coordination: str
    timing_budget_id: str
    material_derivations: tuple[int | None, ...]


def _single(pitch: int) -> tuple[int, ...]:
    return (pitch,)


def _chord(pitch: int) -> tuple[int, ...]:
    return (pitch - 12, pitch - 5, pitch)


_RECIPES = (
    _FixtureRecipe(
        "binary-aba-aligned",
        "stf-bin",
        (1, 1, 2, 1, 4, 2, 1, 1),
        (
            _chord(60),
            _single(62),
            _chord(64),
            _single(65),
            _chord(67),
            _single(65),
            _chord(64),
            _single(62),
            _chord(60),
        ),
        (0, 2, 1, 3, 4, 6, 5, 7, 8),
        "inside",
        ("attack-30ms", "rolled-merge-60ms", "rolled-separate"),
        ("neutral", "neutral", "neutral", "neutral"),
        ("subtle", "subtle", "subtle", "subtle"),
        "aligned",
        "subtle-v1",
        (None, None, None, 0),
    ),
    _FixtureRecipe(
        "dotted-sectional-rubato",
        "stf-dot",
        (2, 3, 2, 4, 6, 3, 2, 4),
        (
            _single(72),
            _chord(69),
            _single(71),
            _chord(68),
            _single(74),
            _chord(71),
            _single(73),
            _chord(70),
            _single(72),
        ),
        (0, 1, 3, 2, 5, 4, 7, 6, 8),
        "inside",
        ("attack-30ms", "rolled-merge-60ms", "rolled-separate"),
        ("savor", "flow", "build", "release"),
        ("moderate", "moderate", "moderate", "moderate"),
        "aligned",
        "narrative-v1",
        (None, 0, None, 0),
    ),
    _FixtureRecipe(
        "rolled-nested-recurrence",
        "stf-roll",
        (4, 9, 8, 6, 16, 9, 12, 4),
        (
            _chord(55),
            _chord(57),
            _single(60),
            _chord(62),
            _chord(59),
            _single(64),
            _chord(60),
            _chord(57),
            _chord(55),
        ),
        (0, 2, 4, 1, 3, 5, 7, 6, 8),
        "inside",
        ("rolled-merge-60ms",),
        ("savor", "neutral", "build", "flow"),
        ("moderate", "moderate", "moderate", "moderate"),
        "rolled",
        "narrative-v2",
        (None, None, None, 0),
    ),
    _FixtureRecipe(
        "irregular-outside-grid",
        "stf-irr",
        (5, 7, 11, 5, 7, 11, 5, 7),
        (
            _single(65),
            _chord(61),
            _single(68),
            _chord(63),
            _single(70),
            _chord(66),
            _single(69),
            _chord(64),
            _single(65),
        ),
        (0, 3, 1, 4, 2, 6, 5, 7, 8),
        "outside",
        ("attack-30ms", "rolled-merge-60ms", "rolled-separate"),
        ("neutral", "neutral", "neutral", "neutral"),
        ("subtle", "subtle", "subtle", "subtle"),
        "aligned",
        "subtle-v1",
        (None, None, None, None),
    ),
)


def _pitch_cycles(recipe: _FixtureRecipe) -> tuple[tuple[tuple[int, ...], ...], ...]:
    transposed = tuple(tuple(pitch + 5 for pitch in group) for group in recipe.template)
    reordered = tuple(recipe.template[index] for index in recipe.permutation)
    return recipe.template, transposed, reordered, recipe.template


def _materials(recipe: _FixtureRecipe) -> tuple[ScoreMaterial, ...]:
    cycle_length = sum(recipe.intervals)
    positions = (0, *tuple(sum(recipe.intervals[:index]) for index in range(1, 9)))
    materials: list[ScoreMaterial] = []
    for block_index, groups in enumerate(_pitch_cycles(recipe)):
        selected = groups if block_index == 3 else groups[:8]
        notes: list[ScoreNote] = []
        for group_index, (at_units, pitches) in enumerate(
            zip(positions[: len(selected)], selected, strict=True)
        ):
            for note_index, pitch in enumerate(pitches):
                notes.append(
                    ScoreNote(
                        event_id=(f"{recipe.prefix}-m{block_index}-g{group_index}-n{note_index}"),
                        at_units=at_units,
                        duration_units=1,
                        pitch=pitch,
                        voice="upper" if pitch == max(pitches) else "lower",
                    )
                )
        derived_index = recipe.material_derivations[block_index]
        materials.append(
            ScoreMaterial(
                material_id=f"{recipe.prefix}-material-{block_index}",
                length_units=cycle_length + (1 if block_index == 3 else 0),
                notes=tuple(notes),
                derived_from=(
                    None if derived_index is None else f"{recipe.prefix}-material-{derived_index}"
                ),
            )
        )
    return tuple(materials)


def _leaf(
    recipe: _FixtureRecipe,
    suffix: str,
    parent_id: str,
    order: int,
    material_index: int,
    role: str,
    *,
    derived_from: str | None = None,
) -> PlanNode:
    return PlanNode(
        node_id=f"{recipe.prefix}-{suffix}",
        parent_id=parent_id,
        order=order,
        role=role,
        derived_from=derived_from,
        duration_weight=1,
        score_material_id=f"{recipe.prefix}-material-{material_index}",
    )


def _plan(recipe: _FixtureRecipe) -> tuple[PiecePlan, tuple[str, ...]]:
    root = f"{recipe.prefix}-root"
    if recipe.fixture_id == "binary-aba-aligned":
        a = _leaf(recipe, "a", root, 0, 0, "opening")
        b = PlanNode(f"{recipe.prefix}-b", root, 1, "contrast")
        b1 = _leaf(recipe, "b1", b.node_id, 0, 1, "contrast")
        b2 = _leaf(recipe, "b2", b.node_id, 1, 2, "variation")
        a2 = _leaf(recipe, "a2", root, 2, 3, "return", derived_from=a.node_id)
        nodes = (PlanNode(root, None, 0, "whole"), a, b, b1, b2, a2)
        leaf_ids = (a.node_id, b1.node_id, b2.node_id, a2.node_id)
    elif recipe.fixture_id == "rolled-nested-recurrence":
        a = PlanNode(f"{recipe.prefix}-a", root, 0, "opening")
        inner_a = _leaf(recipe, "inner-a", a.node_id, 0, 0, "statement")
        inner_b = _leaf(recipe, "inner-b", a.node_id, 1, 1, "variation")
        b = _leaf(recipe, "b", root, 1, 2, "contrast")
        a2 = _leaf(recipe, "a2", root, 2, 3, "return", derived_from=a.node_id)
        nodes = (PlanNode(root, None, 0, "whole"), a, inner_a, inner_b, b, a2)
        leaf_ids = (inner_a.node_id, inner_b.node_id, b.node_id, a2.node_id)
    else:
        roles = ("opening", "contrast", "variation", "return")
        leaves: list[PlanNode] = []
        for index, role in enumerate(roles):
            derived = None
            if index == 3 and recipe.fixture_id == "dotted-sectional-rubato":
                derived = f"{recipe.prefix}-section-0"
            actual_role = role if derived is not None or role != "return" else "release"
            leaves.append(
                _leaf(
                    recipe,
                    f"section-{index}",
                    root,
                    index,
                    index,
                    actual_role,
                    derived_from=derived,
                )
            )
        nodes = (PlanNode(root, None, 0, "whole"), *leaves)
        leaf_ids = tuple(node.node_id for node in leaves)
    return (
        PiecePlan(
            plan_id=f"{recipe.prefix}-plan-v1",
            title=recipe.fixture_id,
            tonal_center=0,
            mode="major",
            root_node_id=root,
            ending_intent="tonic",
            nodes=nodes,
        ),
        leaf_ids,
    )


def _build_fixture(recipe: _FixtureRecipe, output_root: Path) -> KnownTimingFixture:
    plan, leaf_ids = _plan(recipe)
    score = ScoreSpec(
        score_id=f"{recipe.prefix}-score-v1",
        divisions=12,
        materials=_materials(recipe),
    )
    performance = PerformanceSpec(
        performance_id=f"{recipe.prefix}-performance-v1",
        target_duration_ms=24_000,
        default_velocity=64,
        timing_budget_id=recipe.timing_budget_id,
        node_performances=tuple(
            NodePerformance(
                node_id=node_id,
                timing_profile=profile,
                timing_amount=amount,
                dynamics_profile="steady",
                articulation_profile="score",
                coordination_profile=recipe.coordination,
                pedal_profile="none",
            )
            for node_id, profile, amount in zip(
                leaf_ids, recipe.profiles, recipe.amounts, strict=True
            )
        ),
    )
    root = Path(output_root) / recipe.fixture_id
    source = KnownTimingSource(
        case_id=recipe.fixture_id,
        group_id=f"known-fixture-{recipe.fixture_id}",
        root=root,
        piece_path=root / "piece-plan.dsl",
        score_path=root / "score-spec.dsl",
        performance_path=root / "performance-spec.dsl",
        smf_path=root / "final.mid",
        evidence_kind="calibration_result",
        evidence_path=root / "result.json",
        oracle_basis="pitch_rank_constructed",
        oracle_observation_dependency=("present" if recipe.coordination == "rolled" else "absent"),
    )
    fixture = KnownTimingFixture(
        fixture_id=recipe.fixture_id,
        expected_score_grid_membership=recipe.expected_score_grid_membership,
        expected_grouping_profiles=recipe.expected_grouping_profiles,
        plan=plan,
        score=score,
        performance=performance,
        source=source,
    )
    validate_pipeline(plan, score, performance)
    return fixture


def build_known_timing_fixtures(output_root: Path) -> tuple[KnownTimingFixture, ...]:
    """保存先に対応する4つの既知生成元定義を返す。"""
    return tuple(_build_fixture(recipe, Path(output_root)) for recipe in _RECIPES)


def _normalized_intervals(plan: PiecePlan, score: ScoreSpec) -> tuple[int, ...]:
    leaves, _ = ordered_leaf_schedule(plan, score)
    materials = {material.material_id: material for material in score.materials}
    positions = tuple(
        sorted(
            {
                start + note.at_units
                for leaf, start, _ in leaves
                for note in materials[leaf.score_material_id].notes
            }
        )
    )
    intervals = tuple(right - left for left, right in pairwise(positions))
    divisor = math.gcd(*intervals)
    return tuple(value // divisor for value in intervals)


def _tree_shape(plan: PiecePlan) -> str:
    children: dict[str, list[PlanNode]] = {node.node_id: [] for node in plan.nodes}
    for node in plan.nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node)

    def visit(node_id: str) -> object:
        ordered = sorted(children[node_id], key=lambda node: node.order)
        return "leaf" if not ordered else [visit(node.node_id) for node in ordered]

    return json.dumps(visit(plan.root_node_id), separators=(",", ":"))


def _performance_shape(plan: PiecePlan, score: ScoreSpec, performance: PerformanceSpec) -> str:
    leaves, _ = ordered_leaf_schedule(plan, score)
    nodes = {node.node_id: node for node in plan.nodes}
    settings = {item.node_id: item for item in performance.node_performances}
    rows = []
    for leaf, _, _ in leaves:
        rows.append(
            tuple(
                resolve_effective_profile(leaf, nodes, settings, attribute, default)[1]
                for attribute, default in (
                    ("timing_profile", "neutral"),
                    ("timing_amount", "subtle"),
                    ("coordination_profile", "score"),
                )
            )
        )
    return json.dumps(rows, separators=(",", ":"))


def _recurrence_shape(plan: PiecePlan, score: ScoreSpec) -> str:
    node_indexes = {node.node_id: index for index, node in enumerate(plan.nodes)}
    material_indexes = {
        material.material_id: index for index, material in enumerate(score.materials)
    }
    payload = {
        "nodes": [
            [node_indexes[node.derived_from], node_indexes[node.node_id]]
            for node in plan.nodes
            if node.derived_from is not None
        ],
        "materials": [
            [material_indexes[material.derived_from], material_indexes[material.material_id]]
            for material in score.materials
            if material.derived_from is not None
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def fixture_family_fingerprint(fixture: KnownTimingFixture) -> tuple[str, ...]:
    """絶対IDや移調を除いた5成分の設計族fingerprintを返す。"""
    return (
        _tree_shape(fixture.plan),
        json.dumps(_normalized_intervals(fixture.plan, fixture.score), separators=(",", ":")),
        json.dumps(fixture.expected_grouping_profiles, separators=(",", ":")),
        _performance_shape(fixture.plan, fixture.score, fixture.performance),
        _recurrence_shape(fixture.plan, fixture.score),
    )


def known_source_family_fingerprint(source: KnownTimingSource) -> tuple[str, ...]:
    """保存済み既知生成元を人工fixtureと同じ5成分へ変換する。"""
    plan = parse_piece_plan(source.piece_path.read_text(encoding="utf-8"))
    score = parse_score_spec(source.score_path.read_text(encoding="utf-8"))
    performance = parse_performance_spec(source.performance_path.read_text(encoding="utf-8"))
    observed_smf = load_observed_smf(source.smf_path)
    observed_performance = build_observed_performance(observed_smf)
    timing_evidence = align_known_score_timing(plan, score, observed_smf, observed_performance)
    if timing_evidence.note_on_status != "assessed":
        raise ValueError(f"known source note-on alignment failed: {source.case_id}")
    known_units = _known_units(timing_evidence)
    compatible_profiles = tuple(
        profile_id
        for profile_id, groups in build_grouping_profiles(observed_performance).items()
        if compare_grouping_to_known_score(groups, known_units)["status"] == "compatible"
    )
    return (
        _tree_shape(plan),
        json.dumps(_normalized_intervals(plan, score), separators=(",", ":")),
        json.dumps(compatible_profiles, separators=(",", ":")),
        _performance_shape(plan, score, performance),
        _recurrence_shape(plan, score),
    )


def _known_units(timing_evidence: object) -> dict[str, int]:
    return {
        event_id: group.score_unit
        for group in timing_evidence.attack_groups
        for event_id in group.evidence_event_ids
    }


def _compatible_score_grids(intervals: tuple[int, ...]) -> tuple[str, ...]:
    compatible = []
    for grid_id, vocabulary in VOCABULARIES.items():
        scales = {Fraction(intervals[0], 1) / token for token in vocabulary}
        if any(
            all(Fraction(interval, 1) / scale in vocabulary for interval in intervals)
            for scale in scales
        ):
            compatible.append(grid_id)
    return tuple(compatible)


def _assess_fixture(fixture: KnownTimingFixture, temporary_root: Path) -> dict[str, Any]:
    piece_text = dump_piece_plan(fixture.plan)
    score_text = dump_score_spec(fixture.score)
    performance_text = dump_performance_spec(fixture.performance)
    if parse_piece_plan(piece_text) != fixture.plan:
        raise ValueError(f"PiecePlan DSL roundtrip failed: {fixture.fixture_id}")
    if parse_score_spec(score_text) != fixture.score:
        raise ValueError(f"ScoreSpec DSL roundtrip failed: {fixture.fixture_id}")
    if parse_performance_spec(performance_text) != fixture.performance:
        raise ValueError(f"PerformanceSpec DSL roundtrip failed: {fixture.fixture_id}")

    rendered = render_performance(fixture.plan, fixture.score, fixture.performance)
    smf_path = temporary_root / f"{fixture.fixture_id}.mid"
    render_performance_smf(rendered, smf_path)
    observed_smf = load_observed_smf(smf_path)
    observed_performance = build_observed_performance(observed_smf)
    timing_evidence = align_known_score_timing(
        fixture.plan,
        fixture.score,
        observed_smf,
        observed_performance,
    )
    if timing_evidence.note_on_status != "assessed":
        raise ValueError(f"known note-on alignment failed: {fixture.fixture_id}")
    known_units = _known_units(timing_evidence)
    profiles = build_grouping_profiles(observed_performance)
    grouping_diagnostics = {
        profile_id: compare_grouping_to_known_score(groups, known_units)
        for profile_id, groups in profiles.items()
    }
    compatible_profiles = tuple(
        profile_id
        for profile_id, diagnostic in grouping_diagnostics.items()
        if diagnostic["status"] == "compatible"
    )
    candidate_set = generate_score_timing_candidates(
        observed_performance,
        source_ledger_sha256=observed_smf.ledger_sha256,
    )
    comparisons = [
        (
            candidate.candidate_id,
            compare_candidate_to_known_score(
                candidate,
                profiles[candidate.grouping_profile_id],
                known_units,
            ),
        )
        for candidate in candidate_set.candidates
    ]
    equivalent_count = sum(item["status"] == "equivalent" for _, item in comparisons)
    best_match_rate = max(
        (
            float(item["best_scale_interval_match_rate"])
            for _, item in comparisons
            if "best_scale_interval_match_rate" in item
        ),
        default=0.0,
    )
    best_matches = [
        {
            "candidate_id": candidate_id,
            "best_scale_interval_match_rate": best_match_rate,
            "best_scale": item.get("best_scale"),
            "mismatched_intervals": item.get("mismatched_intervals", []),
        }
        for candidate_id, item in comparisons
        if item.get("best_scale_interval_match_rate") == best_match_rate
    ]
    score_position_result = "recovered" if equivalent_count else "not_recovered"
    known_normalized_intervals = _normalized_intervals(fixture.plan, fixture.score)
    compatible_score_grids = _compatible_score_grids(known_normalized_intervals)
    score_positions = tuple(group.score_unit for group in timing_evidence.attack_groups)
    true_map = performance_time_map(fixture.plan, fixture.score, fixture.performance)
    true_times_us = tuple(true_map[position] * 1_000 for position in score_positions)
    time_map_fits = {
        str(count): {
            "boundaries": list(fit.boundaries),
            "mean_absolute_error_us": round(float(fit.anchor_mean_absolute_error), 3),
            "maximum_absolute_error_us": round(float(fit.anchor_maximum_absolute_error), 3),
        }
        for count in (1, 4, 8)
        for fit in (
            fit_equal_piecewise_time_map(
                score_positions,
                true_times_us,
                requested_segment_count=count,
            ),
        )
    }
    return {
        "fixture": fixture,
        "piece_text": piece_text,
        "score_text": score_text,
        "performance_text": performance_text,
        "smf_bytes": smf_path.read_bytes(),
        "lineage": rendered.lineage,
        "note_count": len(rendered.notes),
        "note_on_alignment_status": timing_evidence.note_on_status,
        "attack_group_count": len(timing_evidence.attack_groups),
        "compatible_grouping_profiles": compatible_profiles,
        "grouping_result": "compatible" if compatible_profiles else "incompatible",
        "score_position_result": score_position_result,
        "equivalent_candidate_count": equivalent_count,
        "candidate_count": len(candidate_set.candidates),
        "best_scale_interval_match_rate": best_match_rate,
        "best_matches": best_matches,
        "known_normalized_intervals": known_normalized_intervals,
        "compatible_score_grids": compatible_score_grids,
        "time_map_fits": time_map_fits,
    }


def _write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def write_known_timing_fixtures(
    output_dir: Path,
    *,
    existing_sources: tuple[KnownTimingSource, ...] = (),
) -> dict[str, Any]:
    """全fixtureを一時領域で検証してから、再利用可能な成果物として保存する。"""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    fixtures = build_known_timing_fixtures(output_dir)
    fingerprints = {fixture.fixture_id: fixture_family_fingerprint(fixture) for fixture in fixtures}
    for left, right in combinations(fixtures, 2):
        difference_count = sum(
            a != b
            for a, b in zip(
                fingerprints[left.fixture_id], fingerprints[right.fixture_id], strict=True
            )
        )
        if difference_count < 3:
            raise ValueError(
                f"fixture families differ on only {difference_count} components: "
                f"{left.fixture_id}, {right.fixture_id}"
            )

    existing_by_group: dict[str, KnownTimingSource] = {}
    for source in sorted(existing_sources, key=lambda item: item.case_id):
        existing_by_group.setdefault(source.group_id, source)
    existing_fingerprints = {
        group_id: known_source_family_fingerprint(source)
        for group_id, source in sorted(existing_by_group.items())
    }
    family_comparisons = []
    for fixture in fixtures:
        for group_id, existing in existing_fingerprints.items():
            difference_count = sum(
                left != right
                for left, right in zip(fingerprints[fixture.fixture_id], existing, strict=True)
            )
            family_comparisons.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "existing_group_id": group_id,
                    "different_component_count": difference_count,
                }
            )
            if difference_count < 3:
                raise ValueError(
                    f"fixture and existing family differ on only {difference_count} components: "
                    f"{fixture.fixture_id}, {group_id}"
                )

    with tempfile.TemporaryDirectory(prefix="score-timing-fixtures-") as temporary:
        assessments = [_assess_fixture(fixture, Path(temporary)) for fixture in fixtures]
    for assessment in assessments:
        fixture = assessment["fixture"]
        if tuple(assessment["compatible_grouping_profiles"]) != (
            fixture.expected_grouping_profiles
        ):
            raise ValueError(f"grouping expectation mismatch: {fixture.fixture_id}")
        observed_membership = "inside" if assessment["compatible_score_grids"] else "outside"
        if observed_membership != fixture.expected_score_grid_membership:
            raise ValueError(f"score-grid membership expectation mismatch: {fixture.fixture_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for assessment in assessments:
        fixture = assessment.pop("fixture")
        root = fixture.source.root
        root.mkdir(parents=True)
        _write_text(fixture.source.piece_path, assessment.pop("piece_text"))
        _write_text(fixture.source.score_path, assessment.pop("score_text"))
        _write_text(fixture.source.performance_path, assessment.pop("performance_text"))
        atomic_write_bytes(fixture.source.smf_path, assessment.pop("smf_bytes"))
        evidence = {
            "schema_version": 1,
            "status": "passed",
            "fixture_id": fixture.fixture_id,
            "lineage": list(assessment["lineage"]),
        }
        atomic_write_json(fixture.source.evidence_path, evidence)
        hashes = {
            "piece": sha256_file(fixture.source.piece_path),
            "score": sha256_file(fixture.source.score_path),
            "performance": sha256_file(fixture.source.performance_path),
            "smf": sha256_file(fixture.source.smf_path),
        }
        record = {
            "fixture_id": fixture.fixture_id,
            "group_id": fixture.source.group_id,
            "expected_score_grid_membership": fixture.expected_score_grid_membership,
            **assessment,
            "lineage": list(assessment["lineage"]),
            "family_fingerprint": list(fingerprints[fixture.fixture_id]),
            "hashes": hashes,
        }
        records.append(record)
    atomic_write_bytes(output_dir / "fixtures.jsonl", _jsonl_bytes(records))
    atomic_write_json(
        output_dir / "family-comparisons.json",
        {
            "schema_version": 1,
            "status": "passed",
            "existing_group_count": len(existing_fingerprints),
            "comparisons": family_comparisons,
        },
    )
    result = {
        "schema_version": 1,
        "status": "passed",
        "fixture_count": len(records),
        "inside_count": sum(
            record["expected_score_grid_membership"] == "inside" for record in records
        ),
        "outside_count": sum(
            record["expected_score_grid_membership"] == "outside" for record in records
        ),
    }
    atomic_write_json(output_dir / "result.json", result)
    outputs = ("family-comparisons.json", "fixtures.jsonl", "result.json")
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "outputs": {name: sha256_file(output_dir / name) for name in outputs},
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return result
