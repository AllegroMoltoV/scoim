from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path

from llm_musical_composer.performance_pipeline import (
    PerformanceSpec,
    PiecePlan,
    ScoreMaterial,
    ScoreSpec,
    render_musicxml,
    render_performance,
    render_performance_smf,
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
from llm_musical_composer.recurrence_quality import (
    analyze_foreground_dissonance,
    analyze_foreground_variation,
    analyze_material_harmony,
    analyze_rendered_boundary,
    analyze_rendered_foreground_dissonance,
    analyze_rendered_harmony,
    analyze_section_pacing,
    analyze_transition_connection,
)

ROOT = Path(__file__).parents[1]
V6_INPUT_DIR = ROOT / ".appendix" / "multiscale-calibration-run-v6" / "inputs"
V7_INPUT_DIR = ROOT / ".appendix" / "multiscale-calibration-run-v7" / "inputs"
OUTPUT_DIR = ROOT / ".appendix" / "multiscale-calibration-run-v8"
REPAIR_SCHEME_ID = "multiscale-v8-functional-filter-v1"
EXPECTED_PASSING = 1
EXPECTED_NEIGHBOR = 10
EXPECTED_REPAIRED = 13


def _load_v7():
    path = ROOT / "scripts" / "build-multiscale-calibration-v7.py"
    spec = importlib.util.spec_from_file_location("multiscale_v7", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V7 = _load_v7()


def _read_inputs(
    directory: Path,
) -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    return (
        parse_piece_plan((directory / "piece-plan.music.py").read_text(encoding="utf-8")),
        parse_score_spec((directory / "score.music.py").read_text(encoding="utf-8")),
        parse_performance_spec(
            (directory / "performance.music.py").read_text(encoding="utf-8")
        ),
    )


def _source_scores() -> tuple[ScoreSpec, ScoreSpec]:
    return _read_inputs(V6_INPUT_DIR)[1], _read_inputs(V7_INPUT_DIR)[1]


def _plan() -> PiecePlan:
    plan = _read_inputs(V7_INPUT_DIR)[0]
    return replace(plan, plan_id="multiscale-plan-v8")


def _performance() -> PerformanceSpec:
    performance = _read_inputs(V7_INPUT_DIR)[2]
    return replace(performance, performance_id="multiscale-performance-v8")


def _repair_material(
    material: ScoreMaterial,
    source: ScoreMaterial,
    unsupported_ids: frozenset[str],
) -> ScoreMaterial:
    source_notes = {note.event_id: note for note in source.notes}
    return replace(
        material,
        notes=tuple(
            replace(
                note,
                pitch=source_notes[note.event_id].pitch,
                duration_units=source_notes[note.event_id].duration_units,
            )
            if note.event_id in unsupported_ids
            else note
            for note in material.notes
        ),
    )


def _score() -> ScoreSpec:
    v6_score, v7_score = _source_scores()
    v6_materials = {material.material_id: material for material in v6_score.materials}
    materials: list[ScoreMaterial] = []
    for material in v7_score.materials:
        if not material.material_id.startswith(("a1-", "a2-")):
            materials.append(material)
            continue
        assessment = analyze_foreground_dissonance(v7_score, material.material_id)
        unsupported_ids = frozenset(
            tone.event_id for tone in assessment.tones if tone.role == "unsupported"
        )
        materials.append(
            _repair_material(
                material,
                v6_materials[material.material_id],
                unsupported_ids,
            )
        )
    return replace(v7_score, score_id="multiscale-score-v8", materials=tuple(materials))


def _write_inputs(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
) -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    input_dir = OUTPUT_DIR / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    plan_path = input_dir / "piece-plan.music.py"
    score_path = input_dir / "score.music.py"
    performance_path = input_dir / "performance.music.py"
    plan_path.write_text(dump_piece_plan(plan) + "\n", encoding="utf-8")
    score_path.write_text(dump_score_spec(score) + "\n", encoding="utf-8")
    performance_path.write_text(
        dump_performance_spec(performance) + "\n",
        encoding="utf-8",
    )
    return (
        parse_piece_plan(plan_path.read_text(encoding="utf-8")),
        parse_score_spec(score_path.read_text(encoding="utf-8")),
        parse_performance_spec(performance_path.read_text(encoding="utf-8")),
    )


def _repair_evidence(
    v6_score: ScoreSpec,
    v7_score: ScoreSpec,
    v8_score: ScoreSpec,
) -> dict[str, object]:
    v6_materials = {material.material_id: material for material in v6_score.materials}
    v7_materials = {material.material_id: material for material in v7_score.materials}
    v8_materials = {material.material_id: material for material in v8_score.materials}
    retained: list[str] = []
    repaired: list[str] = []
    for material_id, material in v7_materials.items():
        if not material_id.startswith(("a1-", "a2-")):
            continue
        v6_notes = {note.event_id: note for note in v6_materials[material_id].notes}
        v7_notes = {note.event_id: note for note in material.notes}
        v8_notes = {note.event_id: note for note in v8_materials[material_id].notes}
        for tone in analyze_foreground_dissonance(v7_score, material_id).tones:
            if tone.role == "unsupported":
                repaired.append(tone.event_id)
                if (
                    v8_notes[tone.event_id].pitch,
                    v8_notes[tone.event_id].duration_units,
                ) != (
                    v6_notes[tone.event_id].pitch,
                    v6_notes[tone.event_id].duration_units,
                ):
                    raise ValueError(f"v8 did not restore unsupported tone: {tone.event_id}")
            else:
                retained.append(tone.event_id)
                if v8_notes[tone.event_id] != v7_notes[tone.event_id]:
                    raise ValueError(f"v8 changed functional tone: {tone.event_id}")
    return {
        "retained_event_ids": sorted(retained),
        "repaired_event_ids": sorted(repaired),
        "retained_count": len(retained),
        "repaired_count": len(repaired),
    }


def main() -> int:
    v6_score, v7_score = _source_scores()
    plan, score, performance = _write_inputs(_plan(), _score(), _performance())
    validate_pipeline(plan, score, performance)
    rendered = render_performance(plan, score, performance)
    repair = _repair_evidence(v6_score, v7_score, score)
    transitions = tuple(
        analyze_transition_connection(
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
        for source, transition, target in V7.TRANSITIONS
    )
    boundaries = tuple(
        analyze_rendered_boundary(
            plan,
            rendered,
            transition_node_id=transition,
            target_node_id=target,
            voice="upper",
            maximum_pedal_gap_ms=0,
        )
        for _, transition, target in V7.TRANSITIONS
    )
    variations = tuple(
        analyze_foreground_variation(score, source, target)
        for source, target in V7.VARIATION_PAIRS
    )
    harmony = tuple(
        analyze_material_harmony(
            score,
            material.material_id,
            low_pitch_boundary=48,
            minimum_low_spacing_semitones=7,
        )
        for material in score.materials
        if material.harmonies
    )
    dissonance = tuple(
        analyze_foreground_dissonance(score, material.material_id)
        for material in score.materials
        if material.material_id.startswith(("a1-", "a2-"))
    )
    rendered_harmony = analyze_rendered_harmony(plan, score, rendered)
    pedal_warning = analyze_rendered_foreground_dissonance(plan, score, rendered)
    pacing = analyze_section_pacing(
        plan,
        score,
        rendered,
        source_node_id="a1",
        target_node_id="a2",
        maximum_target_ratio=V7.MAXIMUM_PACING_RATIO,
    )
    pacing_ratio = pacing.target_ms_per_unit / pacing.source_ms_per_unit
    ending = V7._ending_metrics(rendered)
    passing = sum(item.passing_tones for item in dissonance)
    neighbor = sum(item.neighbor_tones for item in dissonance)
    unsupported = sum(item.unsupported_tones for item in dissonance)

    failures: list[str] = []
    failures.extend(
        f"transition:{item.transition_node_id}" for item in transitions if not item.passes
    )
    failures.extend(
        f"rendered-boundary:{item.transition_node_id}" for item in boundaries if not item.passes
    )
    failures.extend(
        f"foreground-variation:{item.target_material_id}" for item in variations if not item.passes
    )
    failures.extend(
        f"harmony:{item.material_id}"
        for item in harmony
        if (
            not item.passes
            or item.accompaniment_chord_tone_ratio != 1.0
            or item.low_spacing_violations
        )
    )
    if (passing, neighbor, unsupported) != (
        EXPECTED_PASSING,
        EXPECTED_NEIGHBOR,
        0,
    ):
        failures.append("functional-dissonance")
    if repair["retained_count"] != EXPECTED_PASSING + EXPECTED_NEIGHBOR:
        failures.append("retained-functional-count")
    if repair["repaired_count"] != EXPECTED_REPAIRED:
        failures.append("repaired-unsupported-count")
    if not rendered_harmony.passes:
        failures.append("rendered-harmony")
    if V7._material_note_counts(score) != V7._material_note_counts(v7_score):
        failures.append("existing-material-note-count")
    if not V7.MINIMUM_PACING_RATIO <= pacing_ratio <= V7.MAXIMUM_PACING_RATIO:
        failures.append("pacing-range")
    if rendered.duration_ms != 180_000:
        failures.append("duration")
    if int(ending["final_note_count"]) != 5:
        failures.append("final-tonic-note-count")
    if frozenset(ending["final_pitch_classes"]) != V7.TONIC_PITCH_CLASSES:
        failures.append("final-tonic-pitches")
    if int(ending["minimum_note_duration_ms"]) < V7.MINIMUM_ENDING_HOLD_MS:
        failures.append("final-tonic-note-duration")
    if int(ending["pedal_hold_after_attack_ms"]) < V7.MINIMUM_ENDING_HOLD_MS:
        failures.append("final-tonic-pedal-duration")
    if int(ending["prior_key_carryover_count"]) != 0:
        failures.append("ending-key-carryover")
    if failures:
        raise ValueError("v8 calibration gates failed: " + ", ".join(failures))

    output_dir = OUTPUT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    musicxml = render_musicxml(plan, score, output_dir / "score.musicxml")
    smf = render_performance_smf(rendered, output_dir / "final.mid")
    payload = {
        "status": "passed",
        "duration_ms": smf.duration_ms,
        "note_count": smf.note_count,
        "musicxml": str(musicxml.resolve()),
        "smf": str(smf.path.resolve()),
        "lineage": rendered.lineage,
        "base_variation_scheme_id": V7.VARIATION_SCHEME_ID,
        "repair_scheme_id": REPAIR_SCHEME_ID,
        "repair": repair,
        "foreground_dissonance": {
            "passing": passing,
            "neighbor": neighbor,
            "unsupported": unsupported,
            "materials": [asdict(item) for item in dissonance],
        },
        "pedal_overlap_warning": asdict(pedal_warning),
        "transitions": [asdict(item) for item in transitions],
        "rendered_boundaries": [asdict(item) for item in boundaries],
        "foreground_variations": [asdict(item) for item in variations],
        "harmony": {
            "materials": [asdict(item) for item in harmony],
            "pedal_carryover_violations": rendered_harmony.pedal_carryover_violations,
        },
        "pacing": {**asdict(pacing), "target_ratio": pacing_ratio},
        "ending_sustain": ending,
    }
    result_path = OUTPUT_DIR / "result.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(result_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
