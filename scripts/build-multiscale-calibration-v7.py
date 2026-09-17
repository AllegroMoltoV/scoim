"""接続句、再現変奏、解決する旋律和声外音を加えた3分曲を生成する。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    ScoreDirection,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
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
from llm_musical_composer.recurrence_analysis import analyze_recurrences
from llm_musical_composer.recurrence_quality import (
    analyze_boundary_breath,
    analyze_foreground_variation,
    analyze_local_pulse,
    analyze_material_harmony,
    analyze_material_vertical_alignment,
    analyze_material_voice_texture,
    analyze_piano_texture_variety,
    analyze_rendered_boundary,
    analyze_rendered_harmony,
    analyze_section_contrast,
    analyze_section_pacing,
    analyze_transition_connection,
    analyze_vertical_alignment,
)

ROOT = Path(__file__).parents[1]
BASE_INPUT_DIR = ROOT / ".appendix" / "multiscale-calibration-run-v6" / "inputs"
OUTPUT_DIR = ROOT / ".appendix" / "multiscale-calibration-run-v7"
VARIATION_SCHEME_ID = "multiscale-v7-sha256-v1"
VARIATION_PAIRS = (
    ("a1-theme", "a1-return"),
    ("a1-theme-prime", "a1-return-prime"),
    ("a2-theme", "a2-return"),
    ("a2-theme-prime", "a2-return-prime"),
    ("b-theme", "b-theme-prime"),
    ("b-contrast", "b-contrast-prime"),
    ("b-theme", "b-return"),
    ("b-theme-prime", "b-return-prime"),
)
TRANSITIONS = (
    ("a1", "transition-ab", "b"),
    ("b", "transition-ba2", "a2"),
)
MINIMUM_PACING_RATIO = 0.84
MAXIMUM_PACING_RATIO = 0.90
MINIMUM_ENDING_HOLD_MS = 2_000
TONIC_PITCH_CLASSES = frozenset({0, 4, 9})


def _base_inputs() -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    return (
        parse_piece_plan((BASE_INPUT_DIR / "piece-plan.music.py").read_text(encoding="utf-8")),
        parse_score_spec((BASE_INPUT_DIR / "score.music.py").read_text(encoding="utf-8")),
        parse_performance_spec(
            (BASE_INPUT_DIR / "performance.music.py").read_text(encoding="utf-8")
        ),
    )


def _plan() -> PiecePlan:
    base, _, _ = _base_inputs()
    nodes: list[PlanNode] = []
    for node in base.nodes:
        if node.node_id == "b":
            nodes.append(
                PlanNode(
                    "transition-ab",
                    "root",
                    1,
                    "transition",
                    duration_weight=1,
                    score_material_id="transition-ab",
                )
            )
            nodes.append(replace(node, order=2))
        elif node.node_id == "a2":
            nodes.append(
                PlanNode(
                    "transition-ba2",
                    "root",
                    3,
                    "transition",
                    duration_weight=1,
                    score_material_id="transition-ba2",
                )
            )
            nodes.append(replace(node, order=4))
        else:
            nodes.append(node)
    return replace(
        base,
        plan_id="multiscale-calibration-v7",
        title="寄り道しながら帰る",
        nodes=tuple(nodes),
    )


def _chord_pitch_classes(root: int, quality: str) -> frozenset[int]:
    intervals = {
        "major": (0, 4, 7),
        "minor": (0, 3, 7),
        "diminished": (0, 3, 6),
    }[quality]
    return frozenset((root + interval) % 12 for interval in intervals)


def _harmony_at(material: ScoreMaterial, onset: int) -> ScoreHarmony:
    return next(
        harmony
        for harmony in material.harmonies
        if harmony.at_units <= onset < harmony.at_units + harmony.duration_units
    )


def _digest(material_id: str, token: str) -> bytes:
    value = f"{VARIATION_SCHEME_ID}|{material_id}|{token}"
    return hashlib.sha256(value.encode("utf-8")).digest()


def _does_not_overlap(
    material: ScoreMaterial,
    note: ScoreNote,
    *,
    pitch: int,
    duration: int,
) -> bool:
    return not any(
        other.event_id != note.event_id
        and other.voice == note.voice
        and other.pitch == pitch
        and other.at_units < note.at_units + duration
        and note.at_units < other.at_units + other.duration_units
        for other in material.notes
    )


def _melodic_tension(material: ScoreMaterial) -> ScoreMaterial:
    if not material.material_id.startswith(("a1-", "a2-")):
        return material
    foreground = tuple(
        sorted(
            (note for note in material.notes if note.voice == "upper"),
            key=lambda note: (note.at_units, note.pitch),
        )
    )
    candidates: list[tuple[int, int]] = []
    for index, note in enumerate(foreground[:-1]):
        if index < 4:
            continue
        target = foreground[index + 1]
        harmony = _harmony_at(material, note.at_units)
        target_harmony = _harmony_at(material, target.at_units)
        if (
            note.at_units == harmony.at_units
            or harmony.harmony_id != target_harmony.harmony_id
            or target.pitch % 12
            not in _chord_pitch_classes(target_harmony.root_pitch_class, target_harmony.quality)
        ):
            continue
        chord = _chord_pitch_classes(harmony.root_pitch_class, harmony.quality)
        deltas = (1, -1, 2, -2)
        offset = _digest(material.material_id, f"direction-{index}")[0] % len(deltas)
        for step in (*deltas[offset:], *deltas[:offset]):
            pitch = target.pitch + step
            if (
                64 <= pitch <= 83
                and pitch % 12 not in chord
                and _does_not_overlap(material, note, pitch=pitch, duration=2)
            ):
                candidates.append((index, pitch))
                break
    ranked = sorted(
        candidates,
        key=lambda item: _digest(material.material_id, f"candidate-{item[0]}"),
    )
    selected: dict[int, int] = {}
    for index, pitch in ranked:
        if any(abs(index - other) <= 1 for other in selected):
            continue
        selected[index] = pitch
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise ValueError(f"two melodic tension candidates were not found: {material.material_id}")
    replacements = {
        foreground[index].event_id: (pitch, min(2, foreground[index].duration_units))
        for index, pitch in selected.items()
    }
    return replace(
        material,
        notes=tuple(
            replace(
                note,
                pitch=replacements[note.event_id][0],
                duration_units=replacements[note.event_id][1],
            )
            if note.event_id in replacements
            else note
            for note in material.notes
        ),
    )


def _b_recurrence_variation(material: ScoreMaterial) -> ScoreMaterial:
    varied_ids = {
        "b-theme-prime",
        "b-contrast-prime",
        "b-return",
        "b-return-prime",
    }
    if material.material_id not in varied_ids:
        return material
    foreground = tuple(
        sorted(
            (note for note in material.notes if note.voice == "lower"),
            key=lambda note: (note.at_units, note.pitch),
        )
    )
    ranked = sorted(
        range(4, len(foreground)),
        key=lambda index: _digest(material.material_id, f"b-candidate-{index}"),
    )
    selected = ranked[:2]
    replacements: dict[str, tuple[int, tuple[str, ...]]] = {}
    for selection_index, index in enumerate(selected):
        note = foreground[index]
        harmony = _harmony_at(material, note.at_units)
        chord = _chord_pitch_classes(harmony.root_pitch_class, harmony.quality)
        pitches = tuple(
            pitch
            for pitch in range(52, 72)
            if pitch % 12 in chord and pitch != note.pitch
        )
        ordered = sorted(pitches, key=lambda pitch: (abs(pitch - note.pitch), pitch))
        offset = _digest(material.material_id, f"b-pitch-{index}")[0] % len(ordered)
        pitch = ordered[offset]
        articulations = note.articulations
        if selection_index == 0:
            articulations = ("accent",) if "accent" not in articulations else ("staccato",)
        replacements[note.event_id] = (pitch, articulations)
    return replace(
        material,
        notes=tuple(
            replace(
                note,
                pitch=replacements[note.event_id][0],
                articulations=replacements[note.event_id][1],
            )
            if note.event_id in replacements
            else note
            for note in material.notes
        ),
    )


def _keep_b_foreground_inside_harmony(material: ScoreMaterial) -> ScoreMaterial:
    if not material.material_id.startswith("b-"):
        return material
    return replace(
        material,
        notes=tuple(
            replace(
                note,
                duration_units=min(
                    note.duration_units,
                    _harmony_at(material, note.at_units).at_units
                    + _harmony_at(material, note.at_units).duration_units
                    - note.at_units,
                ),
            )
            if note.voice == "lower"
            else note
            for note in material.notes
        ),
    )


def _transition_material(
    material_id: str,
    *,
    upper_pitches: tuple[int, ...],
    progressions: tuple[tuple[int, str], tuple[int, str]],
    lower_pitches: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
    dynamic: str,
) -> ScoreMaterial:
    upper_onsets = (0, 3, 4, 6, 9, 10)
    upper = tuple(
        ScoreNote(
            f"{material_id}-u{index}",
            onset,
            min(2, 12 - onset),
            pitch,
            "upper",
            articulations=("tenuto",) if index in {0, 5} else (),
        )
        for index, (onset, pitch) in enumerate(zip(upper_onsets, upper_pitches, strict=True))
    )
    lower: list[ScoreNote] = []
    for span, (root, fifth, third, tail) in enumerate(lower_pitches):
        start = span * 6
        lower.extend(
            (
                ScoreNote(f"{material_id}-l{span}-root", start, 2, root, "lower"),
                ScoreNote(f"{material_id}-l{span}-fifth", start, 2, fifth, "lower"),
                ScoreNote(f"{material_id}-l{span}-third", start + 2, 2, third, "lower"),
                ScoreNote(f"{material_id}-l{span}-tail", start + 4, 2, tail, "lower"),
            )
        )
    harmonies = tuple(
        ScoreHarmony(f"{material_id}-h{index}", index * 6, 6, root, quality)
        for index, (root, quality) in enumerate(progressions)
    )
    return ScoreMaterial(
        material_id,
        12,
        (*upper, *lower),
        directions=(ScoreDirection(f"{material_id}-dynamic", 0, "dynamic", dynamic),),
        harmonies=harmonies,
        foreground_voice="upper",
    )


def _transitions() -> tuple[ScoreMaterial, ScoreMaterial]:
    return (
        _transition_material(
            "transition-ab",
            upper_pitches=(72, 76, 79, 81, 76, 81),
            progressions=((0, "major"), (9, "minor")),
            lower_pitches=((48, 55, 64, 55), (45, 52, 60, 52)),
            dynamic="mf",
        ),
        _transition_material(
            "transition-ba2",
            upper_pitches=(83, 80, 76, 71, 67, 62),
            progressions=((4, "major"), (7, "major")),
            lower_pitches=((40, 47, 56, 59), (43, 50, 59, 62)),
            dynamic="mp",
        ),
    )


def _score() -> ScoreSpec:
    _, base, _ = _base_inputs()
    materials = tuple(
        _b_recurrence_variation(
            _keep_b_foreground_inside_harmony(_melodic_tension(material))
        )
        for material in base.materials
    )
    return replace(
        base,
        score_id="multiscale-score-v7",
        materials=(*materials, *_transitions()),
    )


def _performance() -> PerformanceSpec:
    _, _, base = _base_inputs()
    transitions = (
        NodePerformance(
            "transition-ab",
            timing_profile="build",
            timing_amount="subtle",
            dynamics_profile="shape",
            articulation_profile="legato",
            coordination_profile="score",
            pedal_profile="harmony_legato",
        ),
        NodePerformance(
            "transition-ba2",
            timing_profile="release",
            timing_amount="subtle",
            dynamics_profile="release",
            articulation_profile="legato",
            coordination_profile="score",
            pedal_profile="harmony_legato",
        ),
    )
    return replace(
        base,
        performance_id="multiscale-performance-v7",
        node_performances=(*base.node_performances, *transitions),
    )


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
    performance_path.write_text(dump_performance_spec(performance) + "\n", encoding="utf-8")
    return (
        parse_piece_plan(plan_path.read_text(encoding="utf-8")),
        parse_score_spec(score_path.read_text(encoding="utf-8")),
        parse_performance_spec(performance_path.read_text(encoding="utf-8")),
    )


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


def _material_note_counts(score: ScoreSpec) -> dict[str, int]:
    return {
        material.material_id: len(material.notes)
        for material in score.materials
        if material.material_id.startswith(("a1-", "a2-", "b-"))
    }


def main() -> int:
    base_plan, base_score, base_performance = _base_inputs()
    plan, score, performance = _write_inputs(_plan(), _score(), _performance())
    validate_pipeline(plan, score, performance)
    rendered = render_performance(plan, score, performance)
    base_rendered = render_performance(base_plan, base_score, base_performance)

    transition_assessments = tuple(
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
        for source, transition, target in TRANSITIONS
    )
    boundary_assessments = tuple(
        analyze_rendered_boundary(
            plan,
            rendered,
            transition_node_id=transition,
            target_node_id=target,
            voice="upper",
            maximum_pedal_gap_ms=0,
        )
        for _, transition, target in TRANSITIONS
    )
    variations = tuple(
        analyze_foreground_variation(score, source, target)
        for source, target in VARIATION_PAIRS
    )
    harmonies = tuple(
        analyze_material_harmony(
            score,
            material.material_id,
            low_pitch_boundary=48,
            minimum_low_spacing_semitones=7,
        )
        for material in score.materials
        if material.harmonies
    )
    rendered_harmony = analyze_rendered_harmony(plan, score, rendered)
    recurrences = analyze_recurrences(
        plan,
        score,
        rendered,
        declared_cues={
            node.node_id: ("motif_head",)
            for node in plan.nodes
            if node.derived_from is not None
        },
    )
    voice_textures = tuple(analyze_material_voice_texture(item) for item in score.materials)
    piano_textures = tuple(analyze_piano_texture_variety(item) for item in score.materials)
    breaths = tuple(
        analyze_boundary_breath(plan, score, node_id, minimum_gap_units=4)
        for node_id in ("a1", "b")
    )
    pacing = analyze_section_pacing(
        plan,
        score,
        rendered,
        source_node_id="a1",
        target_node_id="a2",
        maximum_target_ratio=MAXIMUM_PACING_RATIO,
    )
    pacing_ratio = pacing.target_ms_per_unit / pacing.source_ms_per_unit
    contrast = analyze_section_contrast(plan, score, target_node_id="b")
    a1_alignments = tuple(
        analyze_material_vertical_alignment(item, minimum_ratio=0.40)
        for item in score.materials
        if item.material_id.startswith("a1-")
    )
    a2_alignments = tuple(
        analyze_material_vertical_alignment(item, minimum_ratio=0.70)
        for item in score.materials
        if item.material_id.startswith("a2-")
    )
    section_alignment = analyze_vertical_alignment(
        plan,
        rendered,
        source_node_id="a1",
        target_node_id="a2",
        minimum_target_ratio=0.60,
        minimum_increase=0.25,
    )
    pulse = analyze_local_pulse(
        plan,
        score,
        rendered,
        node_id="a1",
        voice="lower",
        minimum_variation_ratio=1.10,
        maximum_adjacent_ratio=1.25,
    )
    ending = _ending_metrics(rendered)

    failures: list[str] = []
    failures.extend(
        f"transition:{item.transition_node_id}"
        for item in transition_assessments
        if not item.passes
    )
    failures.extend(
        f"rendered-boundary:{item.transition_node_id}"
        for item in boundary_assessments
        if not item.passes
    )
    failures.extend(
        f"foreground-variation:{item.target_material_id}" for item in variations if not item.passes
    )
    for item in harmonies:
        if (
            not item.passes
            or item.accompaniment_chord_tone_ratio != 1.0
            or item.low_spacing_violations
        ):
            failures.append(f"harmony:{item.material_id}")
        expected_non_chord = 2 if item.material_id.startswith(("a1-", "a2-")) else 0
        if (
            item.foreground_non_chord_tones != expected_non_chord
            or item.resolved_foreground_non_chord_tones != expected_non_chord
        ):
            failures.append(f"melodic-tension:{item.material_id}")
    if not rendered_harmony.passes:
        failures.append("rendered-harmony")
    failures.extend(
        f"recurrence:{item.target_node_id}"
        for item in recurrences
        if item.relation_status != "related" or item.exact_surface_copy
    )
    failures.extend(
        f"voice-texture:{item.material_id}" for item in voice_textures if not item.passes
    )
    failures.extend(
        f"piano-texture:{item.material_id}" for item in piano_textures if not item.passes
    )
    failures.extend(f"breath:{item.node_id}" for item in breaths if not item.passes)
    failures.extend(
        f"alignment:{item.material_id}"
        for item in (*a1_alignments, *a2_alignments)
        if not item.passes
    )
    if not section_alignment.passes:
        failures.append("section-alignment")
    if not pulse.passes or pulse.variation_ratio < 1.20:
        failures.append("a1-local-pulse")
    if not contrast.passes:
        failures.append("contrast:b")
    if _material_note_counts(score) != _material_note_counts(base_score):
        failures.append("existing-material-note-count")
    if not MINIMUM_PACING_RATIO <= pacing_ratio <= MAXIMUM_PACING_RATIO:
        failures.append("pacing-range")
    if rendered.duration_ms != base_rendered.duration_ms or rendered.duration_ms != 180_000:
        failures.append("duration")
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
        raise ValueError("v7 calibration gates failed: " + ", ".join(failures))

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
        "variation_scheme_id": VARIATION_SCHEME_ID,
        "transitions": [asdict(item) for item in transition_assessments],
        "rendered_boundaries": [asdict(item) for item in boundary_assessments],
        "foreground_variations": [asdict(item) for item in variations],
        "harmony": {
            "materials": [asdict(item) for item in harmonies],
            "pedal_carryover_violations": rendered_harmony.pedal_carryover_violations,
        },
        "pacing": {**asdict(pacing), "target_ratio": pacing_ratio},
        "local_pulse": asdict(pulse),
        "section_alignment": asdict(section_alignment),
        "ending_sustain": ending,
        "base_note_count": len(base_rendered.notes),
        "new_transition_note_count": sum(
            len(material.notes)
            for material in score.materials
            if material.material_id.startswith("transition-")
        ),
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
