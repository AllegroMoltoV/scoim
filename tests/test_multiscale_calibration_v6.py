from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

from llm_musical_composer.performance_pipeline import render_performance
from llm_musical_composer.recurrence_quality import (
    analyze_local_pulse,
    analyze_material_voice_texture,
    analyze_piano_texture_variety,
    analyze_section_contrast,
    analyze_section_pacing,
)

ROOT = Path(__file__).parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"build-multiscale-calibration-{name}.py"
    spec = importlib.util.spec_from_file_location(f"multiscale_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_generators_accept_a_backward_compatible_pacing_limit() -> None:
    for name in ("v3", "v4", "v5"):
        module = _load(name)
        parameter = inspect.signature(module.main).parameters["maximum_target_ratio"]
        assert parameter.default == pytest.approx(0.85)


def test_v6_rebalances_pacing_strengthens_inner_contrast_and_cleans_ending() -> None:
    v5 = _load("v5")
    v6 = _load("v6")
    v5_plan, v5_score, v5_performance = v5._plan(), v5._score(), v5._performance()
    plan, score, performance = v6._plan(), v6._score(), v6._performance()
    rendered_v5 = render_performance(v5_plan, v5_score, v5_performance)
    rendered = render_performance(plan, score, performance)

    v5_b = tuple(item for item in v5_score.materials if item.material_id.startswith("b-"))
    current_b = tuple(item for item in score.materials if item.material_id.startswith("b-"))
    assert current_b == v5_b
    v5_a_counts = {
        item.material_id: len(item.notes)
        for item in v5_score.materials
        if item.material_id.startswith(("a1-", "a2-"))
    }
    assert {
        item.material_id: len(item.notes)
        for item in score.materials
        if item.material_id.startswith(("a1-", "a2-"))
    } == v5_a_counts

    final_node = next(item for item in plan.nodes if item.node_id == "a2-2-3")
    assert final_node.parent_id == "a2-2"
    assert final_node.score_material_id == "final-tonic"
    assert final_node.role == "release"
    final_material = next(item for item in score.materials if item.material_id == "final-tonic")
    assert analyze_material_voice_texture(final_material).passes
    assert analyze_piano_texture_variety(final_material).passes

    for target in ("a1-1", "a2-1"):
        before = analyze_section_contrast(
            v5_plan,
            v5_score,
            target_node_id=target,
            feature_voice="lower",
            minimum_changed_axes=2,
        )
        after = analyze_section_contrast(
            plan,
            score,
            target_node_id=target,
            feature_voice="lower",
            minimum_changed_axes=2,
        )
        assert after.rhythm_grid_distance >= before.rhythm_grid_distance
        assert after.articulation_ratio_delta > before.articulation_ratio_delta
        assert after.passes
    for section in ("a1", "a2"):
        for phrase in range(3):
            target = f"{section}-{phrase}-1"
            before = analyze_section_contrast(
                v5_plan,
                v5_score,
                target_node_id=target,
                feature_voice="lower",
                minimum_changed_axes=1,
            )
            after = analyze_section_contrast(
                plan,
                score,
                target_node_id=target,
                feature_voice="lower",
                minimum_changed_axes=1,
            )
            assert after.articulation_ratio_delta > before.articulation_ratio_delta
            assert after.passes

    pacing = analyze_section_pacing(
        plan,
        score,
        rendered,
        source_node_id="a1",
        target_node_id="a2",
        maximum_target_ratio=0.90,
    )
    pacing_v5 = analyze_section_pacing(
        v5_plan,
        v5_score,
        rendered_v5,
        source_node_id="a1",
        target_node_id="a2",
        maximum_target_ratio=1.0,
    )
    ratio = pacing.target_ms_per_unit / pacing.source_ms_per_unit
    ratio_v5 = pacing_v5.target_ms_per_unit / pacing_v5.source_ms_per_unit
    assert 0.84 <= ratio <= 0.90
    assert abs(1.0 - ratio) < abs(1.0 - ratio_v5)
    pulse = analyze_local_pulse(
        plan,
        score,
        rendered,
        node_id="a1",
        voice="lower",
        minimum_variation_ratio=1.10,
        maximum_adjacent_ratio=1.25,
    )
    assert pulse.variation_ratio >= 1.20
    assert pulse.passes

    final_notes = tuple(note for note in rendered.notes if note.occurrence_node_id == "a2-2-3")
    final_attack = max(note.at_ms for note in final_notes)
    final_chord = tuple(note for note in final_notes if note.at_ms == final_attack)
    assert len(final_chord) == 5
    assert {note.pitch % 12 for note in final_chord} == {0, 4, 9}
    assert min(note.duration_ms for note in final_chord) >= 2_000
    final_release = min(
        pedal.at_ms
        for pedal in rendered.pedals
        if pedal.occurrence_node_id == "a2-2-3" and pedal.value == 0 and pedal.at_ms >= final_attack
    )
    assert final_release - final_attack >= 2_000

    final_interval = next(item for item in rendered.node_intervals if item[0] == "a2-2-3")
    previous_release = max(
        pedal.at_ms
        for pedal in rendered.pedals
        if pedal.occurrence_node_id == "a2-2-2" and pedal.value == 0
    )
    assert previous_release < final_interval[1]
    assert previous_release < final_interval[2]
    held_at_final_attack = {
        note.pitch % 12
        for note in final_notes
        if note.at_ms <= final_attack
        and (
            note.at_ms + note.duration_ms > final_attack
            or final_interval[1] <= note.at_ms < final_attack
        )
    }
    assert held_at_final_attack == {0, 4, 9}
    assert rendered.duration_ms == rendered_v5.duration_ms == 180_000
