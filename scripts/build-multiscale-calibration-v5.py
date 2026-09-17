"""両手共通ルバートとA内部の小尺度差を加えた3分校正曲を生成する。"""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

from llm_musical_composer.performance_pipeline import (
    PerformanceSpec,
    PiecePlan,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    render_performance,
)
from llm_musical_composer.pipeline_dsl import parse_piece_plan, parse_score_spec
from llm_musical_composer.recurrence_quality import (
    analyze_local_pulse,
    analyze_section_contrast,
)

OUTPUT_DIR = Path(".appendix/multiscale-calibration-run-v5")
V4_PATH = Path(__file__).with_name("build-multiscale-calibration-v4.py")


def _load_v4():
    spec = importlib.util.spec_from_file_location("multiscale_v4", V4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("v4 generator could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V4 = _load_v4()
BASE_PLAN = V4._plan
BASE_SCORE = V4._score
BASE_PERFORMANCE = V4._performance


def _plan() -> PiecePlan:
    plan = BASE_PLAN()
    nodes = []
    for node in plan.nodes:
        contrasts_with = node.contrasts_with
        if node.node_id in {"a1-1", "a2-1"}:
            contrasts_with = f"{node.node_id[:2]}-0"
        if re.fullmatch(r"a[12]-[0-2]-1", node.node_id):
            contrasts_with = f"{node.node_id[:-1]}0"
        nodes.append(replace(node, contrasts_with=contrasts_with))
    return replace(
        plan,
        plan_id="multiscale-calibration-v5",
        title="同じ主題を呼吸で語り直す",
        nodes=tuple(nodes),
    )


def _vary_a_material(material: ScoreMaterial) -> ScoreMaterial:
    if not material.material_id.startswith(("a1-", "a2-")):
        return material
    name = material.material_id[3:]
    is_contrast = name.startswith("contrast")
    is_prime = name.endswith("-prime")
    notes: list[ScoreNote] = []
    for note in material.notes:
        if note.voice != "lower":
            notes.append(note)
            continue
        match = re.search(r"-l(\d+)-(root|open|middle|upper)$", note.event_id)
        if match is None:
            notes.append(note)
            continue
        span = int(match.group(1))
        role = match.group(2)
        at_units = note.at_units
        if is_contrast and role == "middle":
            at_units = span * 10 + (3 if material.material_id.startswith("a1-") else 2)
        accented = (
            (is_contrast and role == "middle")
            or (is_prime and not is_contrast and role == "middle")
            or (is_prime and is_contrast and role == "upper")
        )
        notes.append(
            replace(
                note,
                at_units=at_units,
                articulations=("accent",) if accented else (),
            )
        )
    return replace(material, notes=tuple(notes))


def _score() -> ScoreSpec:
    score = BASE_SCORE()
    return replace(
        score,
        score_id="multiscale-score-v5",
        materials=tuple(_vary_a_material(material) for material in score.materials),
    )


def _performance() -> PerformanceSpec:
    performance = BASE_PERFORMANCE()
    return replace(
        performance,
        performance_id="multiscale-performance-v5",
        timing_budget_id="narrative-v2",
        node_performances=tuple(
            replace(item, timing_profile=None, timing_amount=None)
            if item.node_id in {"a2-0", "a2-1", "a2-2"}
            else item
            for item in performance.node_performances
        ),
    )


def _b_materials(score: ScoreSpec) -> tuple[ScoreMaterial, ...]:
    return tuple(material for material in score.materials if material.material_id.startswith("b-"))


def main(maximum_target_ratio: float = 0.85) -> int:
    V4.OUTPUT_DIR = OUTPUT_DIR
    V4._plan = _plan
    V4._score = _score
    V4._performance = _performance
    result = V4.main(maximum_target_ratio=maximum_target_ratio)

    plan = parse_piece_plan((OUTPUT_DIR / "inputs/piece-plan.music.py").read_text(encoding="utf-8"))
    score = parse_score_spec((OUTPUT_DIR / "inputs/score.music.py").read_text(encoding="utf-8"))
    performance = _performance()
    rendered = render_performance(plan, score, performance)
    base_plan = BASE_PLAN()
    base_score = BASE_SCORE()
    base_performance = BASE_PERFORMANCE()
    base_rendered = render_performance(base_plan, base_score, base_performance)

    pulse_v4 = analyze_local_pulse(
        base_plan,
        base_score,
        base_rendered,
        node_id="a1",
        voice="lower",
        minimum_variation_ratio=1.0,
        maximum_adjacent_ratio=2.0,
    )
    pulse_v5 = analyze_local_pulse(
        plan,
        score,
        rendered,
        node_id="a1",
        voice="lower",
        minimum_variation_ratio=1.10,
        maximum_adjacent_ratio=1.25,
    )
    phrase_contrasts = tuple(
        analyze_section_contrast(
            plan,
            score,
            target_node_id=node_id,
            feature_voice="lower",
            minimum_changed_axes=2,
        )
        for node_id in ("a1-1", "a2-1")
    )
    material_contrasts = tuple(
        analyze_section_contrast(
            plan,
            score,
            target_node_id=f"{section}-{phrase}-1",
            feature_voice="lower",
            minimum_changed_axes=1,
        )
        for section in ("a1", "a2")
        for phrase in range(3)
    )

    base_counts = {
        material.material_id: len(material.notes)
        for material in base_score.materials
        if material.material_id.startswith(("a1-", "a2-"))
    }
    current_counts = {
        material.material_id: len(material.notes)
        for material in score.materials
        if material.material_id.startswith(("a1-", "a2-"))
    }
    failures: list[str] = []
    if _b_materials(score) != _b_materials(base_score):
        failures.append("b-material-regression")
    if current_counts != base_counts:
        failures.append("a-note-count-regression")
    if not pulse_v5.passes:
        failures.append("a1-local-pulse")
    if pulse_v5.variation_ratio - 1 < (pulse_v4.variation_ratio - 1) * 2.5:
        failures.append("a1-pulse-not-clearer-than-v4")
    failures.extend(
        f"phrase-contrast:{item.target_node_id}" for item in phrase_contrasts if not item.passes
    )
    failures.extend(
        f"material-contrast:{item.target_node_id}" for item in material_contrasts if not item.passes
    )
    if failures:
        raise ValueError("v5 hierarchy gates failed: " + ", ".join(failures))

    result_path = OUTPUT_DIR / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["local_pulse"] = {
        "v4_a1_lower": asdict(pulse_v4),
        "v5_a1_lower": asdict(pulse_v5),
        "minimum_excess_variation_multiplier": 2.5,
    }
    payload["inner_hierarchy"] = {
        "phrase_contrasts": [asdict(item) for item in phrase_contrasts],
        "material_contrasts": [asdict(item) for item in material_contrasts],
        "b_materials_unchanged": True,
        "a_material_note_counts_unchanged": True,
    }
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(result_path.resolve())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
