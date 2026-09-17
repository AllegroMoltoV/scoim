from __future__ import annotations

import json
from itertools import combinations, pairwise

from llm_musical_composer.performance_pipeline import (
    ordered_leaf_schedule,
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
from llm_musical_composer.reference_timing_run import run_known_timing_compatibility
from llm_musical_composer.score_timing_known_fixtures import (
    build_known_timing_fixtures,
    fixture_family_fingerprint,
    write_known_timing_fixtures,
)

EXPECTED_INTERVAL_CYCLES = {
    "binary-aba-aligned": (1, 1, 2, 1, 4, 2, 1, 1),
    "dotted-sectional-rubato": (2, 3, 2, 4, 6, 3, 2, 4),
    "rolled-nested-recurrence": (4, 9, 8, 6, 16, 9, 12, 4),
    "irregular-outside-grid": (5, 7, 11, 5, 7, 11, 5, 7),
}


def _attack_positions(fixture: object) -> tuple[int, ...]:
    leaves, _ = ordered_leaf_schedule(fixture.plan, fixture.score)
    materials = {material.material_id: material for material in fixture.score.materials}
    return tuple(
        sorted(
            {
                start + note.at_units
                for leaf, start, _ in leaves
                for note in materials[leaf.score_material_id].notes
            }
        )
    )


def test_four_fixed_families_have_declared_score_intervals_and_membership(tmp_path) -> None:
    fixtures = build_known_timing_fixtures(tmp_path)

    assert [fixture.fixture_id for fixture in fixtures] == list(EXPECTED_INTERVAL_CYCLES)
    assert [fixture.expected_score_grid_membership for fixture in fixtures] == [
        "inside",
        "inside",
        "inside",
        "outside",
    ]
    for fixture in fixtures:
        validate_pipeline(fixture.plan, fixture.score, fixture.performance)
        positions = _attack_positions(fixture)
        assert len(positions) == 33
        assert tuple(right - left for left, right in pairwise(positions)) == (
            EXPECTED_INTERVAL_CYCLES[fixture.fixture_id] * 4
        )


def test_fixture_ir_roundtrips_and_ids_do_not_leak_between_families(tmp_path) -> None:
    fixtures = build_known_timing_fixtures(tmp_path)
    seen_ids: set[str] = set()

    for fixture in fixtures:
        assert parse_piece_plan(dump_piece_plan(fixture.plan)) == fixture.plan
        assert parse_score_spec(dump_score_spec(fixture.score)) == fixture.score
        assert parse_performance_spec(dump_performance_spec(fixture.performance)) == (
            fixture.performance
        )
        ids = {
            fixture.plan.plan_id,
            fixture.score.score_id,
            fixture.performance.performance_id,
            *(node.node_id for node in fixture.plan.nodes),
            *(material.material_id for material in fixture.score.materials),
            *(note.event_id for material in fixture.score.materials for note in material.notes),
        }
        assert not ids & seen_ids
        seen_ids.update(ids)


def test_new_families_differ_on_at_least_three_semantic_components(tmp_path) -> None:
    fixtures = build_known_timing_fixtures(tmp_path)
    fingerprints = {fixture.fixture_id: fixture_family_fingerprint(fixture) for fixture in fixtures}

    for left, right in combinations(fixtures, 2):
        left_parts = fingerprints[left.fixture_id]
        right_parts = fingerprints[right.fixture_id]
        assert sum(a != b for a, b in zip(left_parts, right_parts, strict=True)) >= 3


def test_runner_writes_reusable_known_sources_and_separate_recovery_results(tmp_path) -> None:
    output_dir = tmp_path / "fixtures"
    result = write_known_timing_fixtures(output_dir)

    assert result["status"] == "passed"
    records = [
        json.loads(line)
        for line in (output_dir / "fixtures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 4
    assert all(record["note_on_alignment_status"] == "assessed" for record in records)
    assert all(record["attack_group_count"] == 33 for record in records)
    assert all(record["grouping_result"] == "compatible" for record in records)
    assert [record["score_position_result"] for record in records] == [
        "recovered",
        "recovered",
        "not_recovered",
        "not_recovered",
    ]
    assert [record["expected_score_grid_membership"] for record in records] == [
        "inside",
        "inside",
        "inside",
        "outside",
    ]
    assert records[2]["best_scale_interval_match_rate"] == 0.875
    assert {5, 7, 11} <= set(records[-1]["known_normalized_intervals"])
    fixtures = build_known_timing_fixtures(output_dir)
    for fixture in fixtures:
        assert fixture.source.evidence_kind == "calibration_result"
        assert fixture.source.evidence_path.is_file()
    compatibility = run_known_timing_compatibility(
        sources=tuple(fixture.source for fixture in fixtures),
        output_dir=tmp_path / "compatibility",
    )
    assert compatibility["status"] == "pass"
    assert compatibility["affirmative_evidence_count"] == 4
    assert compatibility["score_timing_candidate_recovered_count"] == 2


def test_runner_is_byte_deterministic_across_output_directories(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_known_timing_fixtures(first)
    write_known_timing_fixtures(second)

    first_files = {
        path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()
    }
    assert first_files == second_files
