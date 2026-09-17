from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from llm_musical_composer.attack_frequency_control import scale_score_resolution
from llm_musical_composer.score_timing_known_fixtures import build_known_timing_fixtures
from llm_musical_composer.structure_equivalence_schema import (
    compact_structure_descriptor,
    compact_structure_descriptor_from_dict,
    run_structure_equivalence_schema,
    score_timing_dependency,
    structure_candidate_set_contract_checks,
)
from llm_musical_composer.structure_identifiability import (
    merge_first_neutral_adjacent_materials,
    neutral_adjacent_leaf_pairs,
)

FIXTURE_ROOT = Path(".appendix/score-timing-known-fixtures-v1")


def _fixture(fixture_id: str):
    return next(
        item for item in build_known_timing_fixtures(FIXTURE_ROOT) if item.fixture_id == fixture_id
    )


def _fraction_pair(value) -> tuple[int, int]:
    return value.numerator, value.denominator


def test_descriptor_normalizes_unit_scale_and_distinguishes_same_span_nodes() -> None:
    fixture = _fixture("binary-aba-aligned")
    original = compact_structure_descriptor(fixture.plan, fixture.score)
    scaled = compact_structure_descriptor(
        fixture.plan,
        scale_score_resolution(fixture.score, factor=2),
    )

    assert scaled == original

    merged = merge_first_neutral_adjacent_materials(
        fixture.plan,
        fixture.score,
        fixture.performance,
    )
    assert merged is not None
    plan, score, _, _ = merged
    descriptor = compact_structure_descriptor(plan, score)
    grouped: dict[tuple[tuple[int, int], tuple[int, int]], list[object]] = {}
    for node in descriptor.nodes:
        span = (_fraction_pair(node.start), _fraction_pair(node.end))
        grouped.setdefault(span, []).append(node)

    same_span_different_kind = [
        nodes
        for nodes in grouped.values()
        if len(nodes) > 1 and len({node.node_kind for node in nodes}) > 1
    ]
    assert same_span_different_kind
    assert all(len({node.local_id for node in nodes}) == len(nodes) for nodes in grouped.values())


def test_descriptor_keeps_node_and_material_recurrence_separate() -> None:
    fixture = _fixture("dotted-sectional-rubato")

    descriptor = compact_structure_descriptor(fixture.plan, fixture.score)

    by_pair: dict[tuple[object, object], set[str]] = {}
    for relation in descriptor.derived_relations:
        by_pair.setdefault((relation.source_span, relation.subject_span), set()).add(
            relation.subject_kind
        )
    assert any(kinds == {"node", "material"} for kinds in by_pair.values())


def test_descriptor_keeps_reused_material_occurrences_without_node_roles() -> None:
    fixture = _fixture("binary-aba-aligned")
    leaves = [node for node in fixture.plan.nodes if node.score_material_id is not None]
    reused_plan = replace(
        fixture.plan,
        nodes=tuple(
            replace(node, score_material_id=leaves[0].score_material_id)
            if node.node_id == leaves[1].node_id
            else node
            for node in fixture.plan.nodes
        ),
    )

    descriptor = compact_structure_descriptor(reused_plan, fixture.score)

    assert max(len(material.occurrence_leaf_ids) for material in descriptor.materials) == 2
    assert all(not hasattr(node, "role") for node in descriptor.nodes)


def test_score_timing_dependency_has_an_honest_oracle_only_state() -> None:
    assert score_timing_dependency(None) == {
        "status": "oracle_only",
        "candidate_ids": [],
        "span_source": "oracle_only",
    }


def test_targeted_merge_lists_every_initial_irregular_boundary() -> None:
    fixture = _fixture("irregular-outside-grid")

    pairs = neutral_adjacent_leaf_pairs(
        fixture.plan,
        fixture.score,
        fixture.performance,
    )

    assert pairs == (
        ("stf-irr-section-0", "stf-irr-section-1"),
        ("stf-irr-section-1", "stf-irr-section-2"),
        ("stf-irr-section-2", "stf-irr-section-3"),
    )


def test_runner_closes_candidates_without_copying_note_or_event_tables(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = run_structure_equivalence_schema(workspace=Path("."), output_dir=first)
    second_result = run_structure_equivalence_schema(workspace=Path("."), output_dir=second)

    assert first_result == second_result
    assert first_result["status"] == "pass"
    assert first_result["fixture_count"] == 4
    assert first_result["converged_fixture_count"] == 4
    assert first_result["candidate_count"] > 4
    assert first_result["adoption_status"] == "pending_repeat_comparison"
    assert first_result["descriptor_hash_verified_count"] == first_result["candidate_count"]
    assert first_result["descriptor_roundtrip_verified_count"] == first_result["candidate_count"]
    assert first_result["contract_check_count"] == first_result["contract_check_pass_count"]
    assert first_result["forbidden_field_violation_count"] == 0

    first_files = {
        path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()
    }
    assert first_files == second_files

    sets = [
        json.loads(line)
        for line in (first / "candidate-sets.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {item["score_timing_dependency"]["status"] for item in sets} == {
        "matched_candidate",
        "outside_candidate_space",
    }
    assert all(item["source_kind"] == "oracle_coverage_fixture" for item in sets)
    assert all(item["search_status"] == "complete" for item in sets)
    assert all(item["candidate_count"] <= item["naive_upper_bound"] for item in sets)
    assert any(len(item["candidate_edges"]) > item["candidate_count"] - 1 for item in sets)
    assert all(all(item["contract_checks"].values()) for item in sets)
    assert all(
        compact_structure_descriptor_from_dict(candidate["descriptor"])
        for item in sets
        for candidate in item["candidates"]
    )
    assert all(
        item["terminal_observation"]["last_attack_score_position_candidates"] for item in sets
    )
    assert all(
        item["terminal_observation"]["terminal_tail_assumption_status"] == "assumed_from_candidate"
        for item in sets
    )

    forbidden = {"notes", "events", "attack_groups", "pitch", "velocity", "duration_units"}

    def keys(value):
        if isinstance(value, dict):
            yield from value
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert forbidden.isdisjoint(set(keys(sets)))

    controls = [
        json.loads(line)
        for line in (first / "negative-controls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(controls) == 4
    assert all(item["pitch_change_detected"] is True for item in controls)
    assert all(item["ledger_changed"] is True for item in controls)
    assert all(item["structure_descriptor_equal"] is True for item in controls)

    original = sets[0]
    expected_timing = original["score_timing_dependency"]

    missing_edges = deepcopy(original)
    missing_edges["candidate_edges"] = []
    checks = structure_candidate_set_contract_checks(
        missing_edges,
        expected_timing_dependency=expected_timing,
    )
    assert checks["all_candidates_are_reachable"] is False

    duplicated_scale = deepcopy(original)
    duplicated_scale["scale_controls"] = [original["scale_controls"][0]] * (
        2 * original["candidate_count"]
    )
    checks = structure_candidate_set_contract_checks(
        duplicated_scale,
        expected_timing_dependency=expected_timing,
    )
    assert checks["scale_controls_are_separate_and_valid"] is False

    fabricated_timing = deepcopy(original)
    fabricated_timing["score_timing_dependency"] = {
        "status": "matched_candidate",
        "candidate_ids": ["fabricated"],
        "span_source": "matched_candidate",
    }
    checks = structure_candidate_set_contract_checks(
        fabricated_timing,
        expected_timing_dependency=expected_timing,
    )
    assert checks["score_timing_dependency_is_valid"] is False

    invalid_terminal = deepcopy(original)
    invalid_terminal["terminal_observation"]["last_attack_score_position_candidates"] = [[-1, 1]]
    invalid_terminal["terminal_observation"]["last_note_off_us"] = None
    checks = structure_candidate_set_contract_checks(
        invalid_terminal,
        expected_timing_dependency=expected_timing,
    )
    assert checks["terminal_states_are_separate"] is False
