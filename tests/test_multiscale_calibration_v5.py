from __future__ import annotations

import importlib.util
from pathlib import Path

from llm_musical_composer.performance_pipeline import (
    render_performance,
)
from llm_musical_composer.recurrence_quality import (
    analyze_local_pulse,
    analyze_section_contrast,
)

ROOT = Path(__file__).parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"build-multiscale-calibration-{name}.py"
    spec = importlib.util.spec_from_file_location(f"multiscale_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_changes_only_a_inner_layers_and_shared_rubato() -> None:
    v4 = _load("v4")
    v5 = _load("v5")
    v4_plan, v4_score, v4_performance = v4._plan(), v4._score(), v4._performance()
    plan, score, performance = v5._plan(), v5._score(), v5._performance()

    v4_b = tuple(item for item in v4_score.materials if item.material_id.startswith("b-"))
    v5_b = tuple(item for item in score.materials if item.material_id.startswith("b-"))
    assert v5_b == v4_b
    v4_counts = {
        item.material_id: len(item.notes)
        for item in v4_score.materials
        if item.material_id.startswith(("a1-", "a2-"))
    }
    assert {
        item.material_id: len(item.notes)
        for item in score.materials
        if item.material_id.startswith(("a1-", "a2-"))
    } == v4_counts

    for target in ("a1-1", "a2-1"):
        contrast = analyze_section_contrast(
            plan,
            score,
            target_node_id=target,
            feature_voice="lower",
            minimum_changed_axes=2,
        )
        assert {"rhythm_grid", "articulation"} <= set(contrast.changed_axes)
        assert contrast.passes
    for section in ("a1", "a2"):
        for phrase in range(3):
            contrast = analyze_section_contrast(
                plan,
                score,
                target_node_id=f"{section}-{phrase}-1",
                feature_voice="lower",
                minimum_changed_axes=1,
            )
            assert "articulation" in contrast.changed_axes
            assert contrast.passes

    rendered_v4 = render_performance(v4_plan, v4_score, v4_performance)
    rendered_v5 = render_performance(plan, score, performance)
    pulse_v4 = analyze_local_pulse(
        v4_plan,
        v4_score,
        rendered_v4,
        node_id="a1",
        voice="lower",
        minimum_variation_ratio=1.0,
        maximum_adjacent_ratio=2.0,
    )
    pulse_v5 = analyze_local_pulse(
        plan,
        score,
        rendered_v5,
        node_id="a1",
        voice="lower",
        minimum_variation_ratio=1.10,
        maximum_adjacent_ratio=1.25,
    )

    assert pulse_v5.variation_ratio - 1 >= (pulse_v4.variation_ratio - 1) * 2.5
    assert pulse_v5.passes
    assert rendered_v5.duration_ms == rendered_v4.duration_ms == 180_000
