"""A1とA2の速度差、小区分差、終止長を再校正した3分曲を生成する。"""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    ScoreDirection,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    render_performance,
)
from llm_musical_composer.pipeline_dsl import parse_piece_plan, parse_score_spec
from llm_musical_composer.recurrence_quality import (
    analyze_local_pulse,
    analyze_material_voice_texture,
    analyze_piano_texture_variety,
    analyze_section_contrast,
    analyze_section_pacing,
)

OUTPUT_DIR = Path(".appendix/multiscale-calibration-run-v6")
V5_PATH = Path(__file__).with_name("build-multiscale-calibration-v5.py")
MINIMUM_PACING_RATIO = 0.84
MAXIMUM_PACING_RATIO = 0.90
MINIMUM_ENDING_HOLD_MS = 2_000
TONIC_PITCH_CLASSES = frozenset({0, 4, 9})


def _load_v5():
    spec = importlib.util.spec_from_file_location("multiscale_v5", V5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("v5 generator could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V5 = _load_v5()
BASE_PLAN = V5._plan
BASE_SCORE = V5._score
BASE_PERFORMANCE = V5._performance


def _plan() -> PiecePlan:
    plan = BASE_PLAN()
    nodes = (
        *plan.nodes,
        PlanNode(
            "a2-2-3",
            "a2-2",
            3,
            "release",
            duration_weight=1,
            score_material_id="final-tonic",
        ),
    )
    return replace(
        plan,
        plan_id="multiscale-calibration-v6",
        title="呼吸を整えて余韻へ帰る",
        nodes=nodes,
    )


def _stronger_articulation(material: ScoreMaterial) -> ScoreMaterial:
    if not material.material_id.startswith(("a1-", "a2-")):
        return material
    is_prime = material.material_id.endswith("-prime")
    if not is_prime:
        return material
    is_contrast = "-contrast-prime" in material.material_id
    notes: list[ScoreNote] = []
    for note in material.notes:
        match = re.search(r"-l\d+-(root|open|middle|upper)$", note.event_id)
        if note.voice != "lower" or match is None:
            notes.append(note)
            continue
        role = match.group(1)
        accented = is_contrast or role in {"middle", "upper"}
        notes.append(replace(note, articulations=("accent",) if accented else ()))
    return replace(material, notes=tuple(notes))


def _ending_lead_in(material: ScoreMaterial) -> ScoreMaterial:
    final_prefixes = ("ending-u6", "ending-l3")
    return replace(
        material,
        length_units=20,
        notes=tuple(
            note for note in material.notes if not note.event_id.startswith(final_prefixes)
        ),
    )


def _final_tonic() -> ScoreMaterial:
    return ScoreMaterial(
        "final-tonic",
        16,
        (
            ScoreNote("final-u0", 0, 2, 69, "upper", articulations=("tenuto",)),
            ScoreNote("final-l0", 0, 2, 45, "lower", articulations=("tenuto",)),
            ScoreNote("final-l0-fifth", 0, 2, 52, "lower", articulations=("tenuto",)),
            ScoreNote("final-u1", 2, 2, 64, "upper", articulations=("tenuto",)),
            ScoreNote("final-u2", 4, 12, 69, "upper", articulations=("tenuto",)),
            ScoreNote("final-u2-third", 4, 12, 64, "upper", articulations=("tenuto",)),
            ScoreNote("final-u2-fifth", 4, 12, 60, "upper", articulations=("tenuto",)),
            ScoreNote("final-l1", 4, 12, 45, "lower", articulations=("tenuto",)),
            ScoreNote("final-l1-octave", 4, 12, 57, "lower", articulations=("tenuto",)),
        ),
        directions=(ScoreDirection("final-dynamic", 0, "dynamic", "p"),),
    )


def _score() -> ScoreSpec:
    score = BASE_SCORE()
    materials: list[ScoreMaterial] = []
    for material in score.materials:
        if material.material_id == "ending":
            materials.append(_ending_lead_in(material))
        else:
            materials.append(_stronger_articulation(material))
    materials.append(_final_tonic())
    return replace(
        score,
        score_id="multiscale-score-v6",
        materials=tuple(materials),
    )


def _performance() -> PerformanceSpec:
    performance = BASE_PERFORMANCE()
    items = (
        *(
            replace(item, timing_amount="subtle") if item.node_id == "a2" else item
            for item in performance.node_performances
        ),
        NodePerformance(
            "a2-2-3",
            articulation_profile="legato",
            pedal_profile="clear",
        ),
    )
    return replace(
        performance,
        performance_id="multiscale-performance-v6",
        node_performances=items,
    )


def _b_materials(score: ScoreSpec) -> tuple[ScoreMaterial, ...]:
    return tuple(material for material in score.materials if material.material_id.startswith("b-"))


def _a_note_counts(score: ScoreSpec) -> dict[str, int]:
    return {
        material.material_id: len(material.notes)
        for material in score.materials
        if material.material_id.startswith(("a1-", "a2-"))
    }


def _ending_metrics(rendered) -> dict[str, object]:
    final_notes = tuple(note for note in rendered.notes if note.occurrence_node_id == "a2-2-3")
    final_attack = max(note.at_ms for note in final_notes)
    final_chord = tuple(note for note in final_notes if note.at_ms == final_attack)
    final_release = min(
        pedal.at_ms
        for pedal in rendered.pedals
        if pedal.occurrence_node_id == "a2-2-3" and pedal.value == 0 and pedal.at_ms >= final_attack
    )
    final_interval = next(
        interval for interval in rendered.node_intervals if interval[0] == "a2-2-3"
    )
    previous_release = max(
        pedal.at_ms
        for pedal in rendered.pedals
        if pedal.occurrence_node_id == "a2-2-2" and pedal.value == 0
    )
    held_pitch_classes = {
        note.pitch % 12
        for note in final_notes
        if note.at_ms <= final_attack
        and (
            note.at_ms + note.duration_ms > final_attack
            or final_interval[1] <= note.at_ms < final_attack
        )
    }
    prior_key_carryover = sum(
        note.at_ms + note.duration_ms > final_attack
        for note in rendered.notes
        if note.occurrence_node_id != "a2-2-3"
    )
    return {
        "node_id": "a2-2-3",
        "final_attack_ms": final_attack,
        "final_note_count": len(final_chord),
        "final_pitch_classes": sorted({note.pitch % 12 for note in final_chord}),
        "minimum_note_duration_ms": min(note.duration_ms for note in final_chord),
        "pedal_release_ms": final_release,
        "pedal_hold_after_attack_ms": final_release - final_attack,
        "previous_pedal_release_ms": previous_release,
        "final_leaf_start_ms": final_interval[1],
        "held_pitch_classes_at_final_attack": sorted(held_pitch_classes),
        "prior_key_carryover_count": prior_key_carryover,
    }


def main(maximum_target_ratio: float = MAXIMUM_PACING_RATIO) -> int:
    V5.OUTPUT_DIR = OUTPUT_DIR
    V5._plan = _plan
    V5._score = _score
    V5._performance = _performance
    result = V5.main(maximum_target_ratio=maximum_target_ratio)

    plan = parse_piece_plan((OUTPUT_DIR / "inputs/piece-plan.music.py").read_text(encoding="utf-8"))
    score = parse_score_spec((OUTPUT_DIR / "inputs/score.music.py").read_text(encoding="utf-8"))
    performance = _performance()
    rendered = render_performance(plan, score, performance)
    base_plan = BASE_PLAN()
    base_score = BASE_SCORE()
    base_performance = BASE_PERFORMANCE()
    base_rendered = render_performance(base_plan, base_score, base_performance)

    pacing = analyze_section_pacing(
        plan,
        score,
        rendered,
        source_node_id="a1",
        target_node_id="a2",
        maximum_target_ratio=maximum_target_ratio,
    )
    base_pacing = analyze_section_pacing(
        base_plan,
        base_score,
        base_rendered,
        source_node_id="a1",
        target_node_id="a2",
        maximum_target_ratio=1.0,
    )
    pacing_ratio = pacing.target_ms_per_unit / pacing.source_ms_per_unit
    base_pacing_ratio = base_pacing.target_ms_per_unit / base_pacing.source_ms_per_unit
    pulse = analyze_local_pulse(
        plan,
        score,
        rendered,
        node_id="a1",
        voice="lower",
        minimum_variation_ratio=1.10,
        maximum_adjacent_ratio=1.25,
    )
    phrase_pairs = tuple(
        (
            analyze_section_contrast(
                base_plan,
                base_score,
                target_node_id=node_id,
                feature_voice="lower",
                minimum_changed_axes=2,
            ),
            analyze_section_contrast(
                plan,
                score,
                target_node_id=node_id,
                feature_voice="lower",
                minimum_changed_axes=2,
            ),
        )
        for node_id in ("a1-1", "a2-1")
    )
    material_pairs = tuple(
        (
            analyze_section_contrast(
                base_plan,
                base_score,
                target_node_id=f"{section}-{phrase}-1",
                feature_voice="lower",
                minimum_changed_axes=1,
            ),
            analyze_section_contrast(
                plan,
                score,
                target_node_id=f"{section}-{phrase}-1",
                feature_voice="lower",
                minimum_changed_axes=1,
            ),
        )
        for section in ("a1", "a2")
        for phrase in range(3)
    )
    ending = _ending_metrics(rendered)
    final_material = next(
        material for material in score.materials if material.material_id == "final-tonic"
    )
    final_voice_texture = analyze_material_voice_texture(final_material)
    final_piano_texture = analyze_piano_texture_variety(final_material)

    failures: list[str] = []
    if _b_materials(score) != _b_materials(base_score):
        failures.append("b-material-regression")
    if _a_note_counts(score) != _a_note_counts(base_score):
        failures.append("a-note-count-regression")
    if not MINIMUM_PACING_RATIO <= pacing_ratio <= maximum_target_ratio:
        failures.append("pacing-range")
    if abs(1.0 - pacing_ratio) >= abs(1.0 - base_pacing_ratio):
        failures.append("pacing-not-closer-than-v5")
    if not pulse.passes or pulse.variation_ratio < 1.20:
        failures.append("a1-local-pulse")
    failures.extend(
        f"phrase-rhythm:{after.target_node_id}"
        for before, after in phrase_pairs
        if after.rhythm_grid_distance < before.rhythm_grid_distance
    )
    failures.extend(
        f"phrase-articulation:{after.target_node_id}"
        for before, after in phrase_pairs
        if after.articulation_ratio_delta <= before.articulation_ratio_delta
    )
    failures.extend(
        f"material-articulation:{after.target_node_id}"
        for before, after in material_pairs
        if after.articulation_ratio_delta <= before.articulation_ratio_delta
    )
    if not final_voice_texture.passes or not final_piano_texture.passes:
        failures.append("final-tonic-texture")
    if int(ending["final_note_count"]) != 5:
        failures.append("final-tonic-note-count")
    if frozenset(ending["final_pitch_classes"]) != TONIC_PITCH_CLASSES:
        failures.append("final-tonic-pitches")
    if int(ending["minimum_note_duration_ms"]) < MINIMUM_ENDING_HOLD_MS:
        failures.append("final-tonic-note-duration")
    if int(ending["pedal_hold_after_attack_ms"]) < MINIMUM_ENDING_HOLD_MS:
        failures.append("final-tonic-pedal-duration")
    if int(ending["previous_pedal_release_ms"]) >= int(ending["final_leaf_start_ms"]):
        failures.append("ending-pedal-boundary")
    if frozenset(ending["held_pitch_classes_at_final_attack"]) != TONIC_PITCH_CLASSES:
        failures.append("ending-held-pitch-classes")
    if int(ending["prior_key_carryover_count"]) != 0:
        failures.append("ending-key-carryover")
    if failures:
        raise ValueError("v6 calibration gates failed: " + ", ".join(failures))

    result_path = OUTPUT_DIR / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["pacing_recalibration"] = {
        "v5": asdict(base_pacing),
        "v5_target_ratio": base_pacing_ratio,
        "v6": asdict(pacing),
        "v6_target_ratio": pacing_ratio,
        "minimum_target_ratio": MINIMUM_PACING_RATIO,
        "maximum_target_ratio": maximum_target_ratio,
    }
    payload["inner_hierarchy_v6"] = {
        "phrase_contrasts": [
            {"v5": asdict(before), "v6": asdict(after)} for before, after in phrase_pairs
        ],
        "material_contrasts": [
            {"v5": asdict(before), "v6": asdict(after)} for before, after in material_pairs
        ],
        "b_materials_unchanged": True,
        "a_material_note_counts_unchanged": True,
    }
    payload["ending_sustain"] = ending
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(result_path.resolve())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
