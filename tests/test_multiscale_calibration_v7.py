from __future__ import annotations

import importlib.util
from pathlib import Path

from llm_musical_composer.performance_pipeline import render_performance
from llm_musical_composer.recurrence_quality import (
    analyze_foreground_variation,
    analyze_material_harmony,
    analyze_rendered_boundary,
    analyze_rendered_harmony,
    analyze_section_pacing,
    analyze_transition_connection,
)

ROOT = Path(__file__).parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"build-multiscale-calibration-{name}.py"
    spec = importlib.util.spec_from_file_location(f"multiscale_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v7_adds_connected_fills_varied_recurrences_and_resolved_melodic_tension() -> None:
    v6 = _load("v6")
    v7 = _load("v7")
    base_score = v6._score()
    plan, score, performance = v7._plan(), v7._score(), v7._performance()
    rendered = render_performance(plan, score, performance)

    transitions = tuple(node for node in plan.nodes if node.role == "transition")
    assert tuple(node.node_id for node in transitions) == ("transition-ab", "transition-ba2")
    assert len({node.score_material_id for node in transitions}) == 2
    for source, transition, target in (
        ("a1", "transition-ab", "b"),
        ("b", "transition-ba2", "a2"),
    ):
        connection = analyze_transition_connection(
            plan,
            score,
            source_node_id=source,
            transition_node_id=transition,
            target_node_id=target,
            voice="upper",
            minimum_transition_attacks=6,
            maximum_boundary_leap=2,
            maximum_internal_leap=5,
        )
        boundary = analyze_rendered_boundary(
            plan,
            rendered,
            transition_node_id=transition,
            target_node_id=target,
            voice="upper",
            maximum_pedal_gap_ms=0,
        )
        assert connection.passes
        assert boundary.pedal_gap_ms == 0
        assert not boundary.same_pitch_restrike
        assert boundary.passes

    for source, target in v7.VARIATION_PAIRS:
        assessment = analyze_foreground_variation(score, source, target)
        assert assessment.foreground_motif_head_preserved
        assert not assessment.exact_surface_copy
        assert assessment.passes

    harmonic = {
        material.material_id: analyze_material_harmony(
            score,
            material.material_id,
            low_pitch_boundary=48,
            minimum_low_spacing_semitones=7,
        )
        for material in score.materials
        if material.harmonies
    }
    for material_id, assessment in harmonic.items():
        assert assessment.unresolved_foreground_non_chord_tones == 0
        assert assessment.accompaniment_chord_tone_ratio == 1.0
        if material_id.startswith(("a1-", "a2-")):
            assert assessment.foreground_non_chord_tones == 2
            assert assessment.resolved_foreground_non_chord_tones == 2
        else:
            assert assessment.foreground_non_chord_tones == 0
    assert analyze_rendered_harmony(plan, score, rendered).passes

    base_counts = {
        item.material_id: len(item.notes)
        for item in base_score.materials
        if item.material_id.startswith(("a1-", "a2-", "b-"))
    }
    assert {
        item.material_id: len(item.notes)
        for item in score.materials
        if item.material_id in base_counts
    } == base_counts
    assert v7._score() == score
    assert rendered.duration_ms == 180_000
    pacing = analyze_section_pacing(
        plan,
        score,
        rendered,
        source_node_id="a1",
        target_node_id="a2",
        maximum_target_ratio=0.90,
    )
    ratio = pacing.target_ms_per_unit / pacing.source_ms_per_unit
    assert 0.84 <= ratio <= 0.90
    assert v7.VARIATION_SCHEME_ID == "multiscale-v7-sha256-v1"
