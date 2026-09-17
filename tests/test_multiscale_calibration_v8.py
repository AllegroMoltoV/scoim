from __future__ import annotations

import importlib.util
from pathlib import Path

from llm_musical_composer.performance_pipeline import (
    render_performance,
    render_performance_smf,
)
from llm_musical_composer.recurrence_quality import (
    analyze_foreground_dissonance,
    analyze_foreground_variation,
    analyze_rendered_boundary,
    analyze_rendered_foreground_dissonance,
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


def _notes_by_id(material):
    return {note.event_id: note for note in material.notes}


def test_v8_keeps_functional_dissonance_and_repairs_only_unsupported_events(tmp_path) -> None:
    v6 = _load("v6")
    v7 = _load("v7")
    v8 = _load("v8")
    v6_score = v6._score()
    v7_score = v7._score()
    plan, score, performance = v8._plan(), v8._score(), v8._performance()
    rendered = render_performance(plan, score, performance)
    v6_materials = {item.material_id: item for item in v6_score.materials}
    v7_materials = {item.material_id: item for item in v7_score.materials}
    v8_materials = {item.material_id: item for item in score.materials}

    retained_ids: set[str] = set()
    repaired_ids: set[str] = set()
    passing = neighbor = unsupported = 0
    for material_id, material in v7_materials.items():
        if not material_id.startswith(("a1-", "a2-")):
            continue
        v7_assessment = analyze_foreground_dissonance(v7_score, material_id)
        v8_assessment = analyze_foreground_dissonance(score, material_id)
        passing += v8_assessment.passing_tones
        neighbor += v8_assessment.neighbor_tones
        unsupported += v8_assessment.unsupported_tones
        v6_notes = _notes_by_id(v6_materials[material_id])
        v7_notes = _notes_by_id(material)
        v8_notes = _notes_by_id(v8_materials[material_id])
        for tone in v7_assessment.tones:
            if tone.role == "unsupported":
                repaired_ids.add(tone.event_id)
                assert (
                    v8_notes[tone.event_id].pitch,
                    v8_notes[tone.event_id].duration_units,
                ) == (
                    v6_notes[tone.event_id].pitch,
                    v6_notes[tone.event_id].duration_units,
                )
            else:
                retained_ids.add(tone.event_id)
                assert v8_notes[tone.event_id] == v7_notes[tone.event_id]

    assert (passing, neighbor, unsupported) == (1, 10, 0)
    assert len(retained_ids) == 11
    assert len(repaired_ids) == 13
    for material_id, material in v7_materials.items():
        if material_id.startswith(("a1-", "a2-")):
            continue
        assert v8_materials[material_id] == material

    for source, transition, target in v7.TRANSITIONS:
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
        assert boundary.passes
    for source, target in v7.VARIATION_PAIRS:
        assert analyze_foreground_variation(score, source, target).passes
    assert analyze_rendered_harmony(plan, score, rendered).passes
    pedal_warning = analyze_rendered_foreground_dissonance(plan, score, rendered)
    assert pedal_warning.pedal_overlap_count > 0
    assert pedal_warning.maximum_overlap_ms > 0

    pacing = analyze_section_pacing(
        plan,
        score,
        rendered,
        source_node_id="a1",
        target_node_id="a2",
        maximum_target_ratio=0.90,
    )
    assert 0.84 <= pacing.target_ms_per_unit / pacing.source_ms_per_unit <= 0.90
    assert rendered.duration_ms == 180_000
    assert v8._score() == score
    first = render_performance_smf(rendered, tmp_path / "first.mid")
    second = render_performance_smf(rendered, tmp_path / "second.mid")
    assert first.path.read_bytes() == second.path.read_bytes()
