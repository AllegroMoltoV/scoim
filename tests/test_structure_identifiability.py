from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from llm_musical_composer.performance_pipeline import (
    ordered_leaf_schedule,
    render_performance,
    validate_pipeline,
)
from llm_musical_composer.score_timing_known_fixtures import build_known_timing_fixtures
from llm_musical_composer.structure_identifiability import (
    compare_structure_variant,
    erase_recurrence_annotations,
    flatten_first_inert_internal_node,
    merge_first_neutral_adjacent_materials,
    run_structure_identifiability,
    scale_divisions_only,
)

FIXTURE_ROOT = Path(".appendix/score-timing-known-fixtures-v1")


def _fixture(fixture_id: str):
    return next(
        item for item in build_known_timing_fixtures(FIXTURE_ROOT) if item.fixture_id == fixture_id
    )


def _musical_notes(rendered):
    return tuple(
        (note.at_ms, note.duration_ms, note.pitch, note.velocity) for note in rendered.notes
    )


def test_recurrence_erasure_changes_annotations_without_changing_performance() -> None:
    fixture = _fixture("dotted-sectional-rubato")

    variant = erase_recurrence_annotations(fixture.plan, fixture.score)

    assert variant is not None
    plan, score = variant
    validate_pipeline(plan, score, fixture.performance)
    assert all(node.derived_from is None for node in plan.nodes)
    assert all(node.role != "return" for node in plan.nodes)
    assert all(material.derived_from is None for material in score.materials)
    assert (
        render_performance(plan, score, fixture.performance).notes
        == render_performance(fixture.plan, fixture.score, fixture.performance).notes
    )


def test_flattening_changes_tree_but_preserves_leaf_schedule_and_performance() -> None:
    fixture = _fixture("binary-aba-aligned")
    original_leaf_ids = tuple(
        node.node_id for node, _, _ in ordered_leaf_schedule(fixture.plan, fixture.score)[0]
    )

    variant = flatten_first_inert_internal_node(fixture.plan, fixture.performance)

    assert variant is not None
    plan, performance, removed_node_id = variant
    validate_pipeline(plan, fixture.score, performance)
    assert removed_node_id == "stf-bin-b"
    assert len(plan.nodes) == len(fixture.plan.nodes) - 1
    assert (
        tuple(node.node_id for node, _, _ in ordered_leaf_schedule(plan, fixture.score)[0])
        == original_leaf_ids
    )
    assert (
        render_performance(plan, fixture.score, performance).notes
        == render_performance(fixture.plan, fixture.score, fixture.performance).notes
    )


def test_material_merge_preserves_sounding_notes_but_changes_provenance() -> None:
    fixture = _fixture("irregular-outside-grid")

    variant = merge_first_neutral_adjacent_materials(
        fixture.plan,
        fixture.score,
        fixture.performance,
    )

    assert variant is not None
    plan, score, performance, merged_leaf_ids = variant
    validate_pipeline(plan, score, performance)
    assert merged_leaf_ids == ("stf-irr-section-0", "stf-irr-section-1")
    assert len(score.materials) == len(fixture.score.materials) - 1
    original = render_performance(fixture.plan, fixture.score, fixture.performance)
    merged = render_performance(plan, score, performance)
    assert _musical_notes(merged) == _musical_notes(original)
    assert merged.notes != original.notes


def test_divisions_only_changes_score_metadata_without_changing_performance() -> None:
    fixture = _fixture("rolled-nested-recurrence")

    scaled = scale_divisions_only(fixture.score, factor=2)

    assert scaled.divisions == fixture.score.divisions * 2
    assert scaled.materials == fixture.score.materials
    original = render_performance(fixture.plan, fixture.score, fixture.performance)
    changed = render_performance(fixture.plan, scaled, fixture.performance)
    assert changed.notes == original.notes
    assert changed.pedals == original.pedals
    assert changed.node_intervals == original.node_intervals
    assert changed.lineage != original.lineage


def test_variant_comparison_separates_smf_meaning_and_provenance(tmp_path: Path) -> None:
    fixture = _fixture("binary-aba-aligned")
    variant = merge_first_neutral_adjacent_materials(
        fixture.plan,
        fixture.score,
        fixture.performance,
    )
    assert variant is not None
    plan, score, performance, _ = variant

    result = compare_structure_variant(
        fixture,
        transform_id="merge-neutral-adjacent",
        variant_plan=plan,
        variant_score=score,
        variant_performance=performance,
        temporary_root=tmp_path,
    )

    assert result["status"] == "exact_smf_collision"
    assert result["smf_bytes_equal"] is True
    assert result["smf_performance_meaning_equal"] is True
    assert result["rendered_notes_literal_equal"] is False
    assert result["performed_provenance_equal"] is False


def test_one_tick_classification_does_not_ignore_pedal_difference(tmp_path: Path) -> None:
    fixture = _fixture("binary-aba-aligned")
    first = fixture.performance.node_performances[0]
    changed_performance = replace(
        fixture.performance,
        node_performances=(
            replace(first, pedal_profile="phrase_legato"),
            *fixture.performance.node_performances[1:],
        ),
    )

    result = compare_structure_variant(
        fixture,
        transform_id="pedal-negative-control",
        variant_plan=fixture.plan,
        variant_score=fixture.score,
        variant_performance=changed_performance,
        temporary_root=tmp_path,
    )

    assert result["maximum_note_time_difference_ms"] == 0
    assert result["rendered_pedals_literal_equal"] is False
    assert result["status"] == "different"


def test_runner_records_fixed_transform_outcomes_and_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = run_structure_identifiability(workspace=Path("."), output_dir=first)
    second_result = run_structure_identifiability(workspace=Path("."), output_dir=second)

    assert first_result == second_result
    assert first_result["status"] == "pass"
    assert first_result["source_fixture_count"] == 4
    assert first_result["artificial_design_family_count"] == 4
    assert first_result["comparison_count"] == 20
    assert first_result["negative_control_detected_count"] == 4
    assert first_result["transform_summary"] == {
        "divisions-only-x2": {
            "applicable_count": 4,
            "exact_smf_collision_count": 4,
            "not_applicable_count": 0,
            "within_one_transport_tick_count": 0,
        },
        "erase-recurrence": {
            "applicable_count": 3,
            "exact_smf_collision_count": 3,
            "not_applicable_count": 1,
            "within_one_transport_tick_count": 0,
        },
        "flatten-inert-node": {
            "applicable_count": 2,
            "exact_smf_collision_count": 2,
            "not_applicable_count": 2,
            "within_one_transport_tick_count": 0,
        },
        "full-score-resolution-x2": {
            "applicable_count": 4,
            "exact_smf_collision_count": 3,
            "not_applicable_count": 0,
            "within_one_transport_tick_count": 1,
        },
        "merge-neutral-adjacent": {
            "applicable_count": 2,
            "exact_smf_collision_count": 2,
            "not_applicable_count": 2,
            "within_one_transport_tick_count": 0,
        },
    }
    first_files = {
        path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()
    }
    assert first_files == second_files
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert {
        "attack_frequency_control.py",
        "performance_pipeline.py",
        "structure_identifiability.py",
    }.issubset(manifest["inputs"])
