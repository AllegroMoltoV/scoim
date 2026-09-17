"""整った帰還形を正本とし、そこから冒頭の崩しを派生させる3分校正曲を生成する。"""

from __future__ import annotations

import json
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
    analyze_material_vertical_alignment,
    analyze_material_voice_texture,
    analyze_piano_texture_variety,
    analyze_section_contrast,
    analyze_section_pacing,
    analyze_vertical_alignment,
)

OUTPUT_DIR = Path(".appendix/multiscale-calibration-run-v3")
ALIGNED_ONSETS = (0, 4, 6, 8, 12, 16, 20, 24, 26, 28, 32, 34)
ALIGNED_BASS_ONSETS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 34)
BROKEN_ONSETS = (0, 5, 8, 12, 15, 19, 22, 26, 29, 31, 33, 34)
BROKEN_BASS_ONSETS = (0, 7, 13, 19, 25, 31, 34)
B_LOWER_ONSETS = (0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 28, 30)
B_UPPER_ONSETS = (0, 8, 16, 24, 30)

# A2が作曲上の正本であり、A1は対応するA2素材から生成する。
CORE_SPECS = (
    (
        "theme",
        69,
        (0, 2, 3, 7, 5, 3, 2, -2, 0, 3, 2, 0),
        (0, 7, 3, 8, 5, 7, 3, 8, 7, 0),
        "mp",
        None,
    ),
    (
        "theme-prime",
        69,
        (0, 2, 3, 7, 8, 7, 5, 3, 2, 5, 3, 0),
        (0, 7, 3, 10, 5, 8, 3, 7, 5, 0),
        "mp",
        "a2-theme",
    ),
    (
        "contrast",
        72,
        (0, 3, 7, 5, 8, 7, 3, 5, 2, 0, 3, 0),
        (0, 7, 3, 10, 5, 8, 7, 3, 5, 0),
        "mf",
        None,
    ),
    (
        "contrast-prime",
        72,
        (0, 3, 7, 5, 3, 0, -2, 2, 5, 3, 2, 0),
        (0, 7, 3, 8, 5, 10, 7, 3, 2, 0),
        "mf",
        "a2-contrast",
    ),
    (
        "return",
        69,
        (0, 2, 3, 7, 3, 5, 8, 7, 2, 0, 2, 0),
        (0, 7, 3, 8, 5, 7, 3, 8, 7, 0),
        "mp",
        "a2-theme",
    ),
    (
        "return-prime",
        69,
        (0, 2, 3, 7, 10, 8, 5, 3, -2, 2, 3, 0),
        (0, 7, 3, 10, 5, 8, 3, 7, 5, 0),
        "p",
        "a2-theme-prime",
    ),
)


def _third_below(pitch: int) -> int:
    for distance in (3, 4):
        candidate = pitch - distance
        if candidate % 12 in {9, 11, 0, 2, 4, 5, 7}:
            return candidate
    return pitch - 3


def _directions(material_id: str, length: int, dynamic: str) -> tuple[ScoreDirection, ...]:
    return (
        ScoreDirection(f"{material_id}-dynamic", 0, "dynamic", dynamic),
        ScoreDirection(f"{material_id}-breath", length - 4, "breath", "light"),
    )


def _aligned_material(
    name: str,
    center: int,
    melody: tuple[int, ...],
    bass: tuple[int, ...],
    dynamic: str,
    derived_from: str | None,
) -> ScoreMaterial:
    material_id = f"a2-{name}"
    notes: list[ScoreNote] = []
    for index, (onset, interval) in enumerate(zip(ALIGNED_ONSETS, melody, strict=True)):
        next_onset = ALIGNED_ONSETS[index + 1] if index + 1 < len(ALIGNED_ONSETS) else 40
        pitch = center + interval
        articulation = ("accent",) if index in {0, 4, 8} else ()
        notes.append(
            ScoreNote(
                f"{material_id}-u{index}",
                onset,
                min(4, next_onset - onset),
                pitch,
                "upper",
                articulations=articulation,
            )
        )
        if onset in {0, 12, 24, 34}:
            notes.append(
                ScoreNote(
                    f"{material_id}-u{index}-third",
                    onset,
                    min(4, next_onset - onset),
                    _third_below(pitch),
                    "upper",
                    articulations=articulation,
                )
            )
    root = center - 24
    for index, (onset, interval) in enumerate(zip(ALIGNED_BASS_ONSETS, bass, strict=True)):
        next_onset = ALIGNED_BASS_ONSETS[index + 1] if index + 1 < len(ALIGNED_BASS_ONSETS) else 40
        pitch = root + interval
        articulation = ("accent",) if index in {0, 4, 8} else ()
        notes.append(
            ScoreNote(
                f"{material_id}-l{index}",
                onset,
                min(5, next_onset - onset),
                pitch,
                "lower",
                articulations=articulation,
            )
        )
        if onset in {0, 16, 34}:
            notes.append(
                ScoreNote(
                    f"{material_id}-l{index}-fifth",
                    onset,
                    min(5, next_onset - onset),
                    pitch + 7,
                    "lower",
                    articulations=articulation,
                )
            )
    return ScoreMaterial(
        material_id,
        40,
        tuple(notes),
        derived_from=derived_from,
        directions=_directions(material_id, 40, dynamic),
    )


def _broken_material(
    name: str,
    center: int,
    melody: tuple[int, ...],
    bass: tuple[int, ...],
    dynamic: str,
) -> ScoreMaterial:
    material_id = f"a1-{name}"
    notes: list[ScoreNote] = []
    for index, (onset, interval) in enumerate(zip(BROKEN_ONSETS, melody, strict=True)):
        next_onset = BROKEN_ONSETS[index + 1] if index + 1 < len(BROKEN_ONSETS) else 40
        pitch = center + interval
        articulation = ("tenuto",) if index in {0, len(melody) - 1} else ()
        notes.append(
            ScoreNote(
                f"{material_id}-u{index}",
                onset,
                min(4, next_onset - onset),
                pitch,
                "upper",
                articulations=articulation,
            )
        )
        if onset in {0, 12, 26, 34}:
            notes.append(
                ScoreNote(
                    f"{material_id}-u{index}-third",
                    onset,
                    min(4, next_onset - onset),
                    _third_below(pitch),
                    "upper",
                    articulations=articulation,
                )
            )
    broken_bass = tuple(bass[index] for index in (0, 2, 3, 5, 6, 8, 9))
    root = center - 24
    for index, (onset, interval) in enumerate(zip(BROKEN_BASS_ONSETS, broken_bass, strict=True)):
        next_onset = BROKEN_BASS_ONSETS[index + 1] if index + 1 < len(BROKEN_BASS_ONSETS) else 40
        pitch = root + interval
        articulation = ("tenuto",) if index in {0, len(broken_bass) - 1} else ()
        notes.append(
            ScoreNote(
                f"{material_id}-l{index}",
                onset,
                min(6, next_onset - onset),
                pitch,
                "lower",
                articulations=articulation,
            )
        )
        if onset in {0, 19, 34}:
            notes.append(
                ScoreNote(
                    f"{material_id}-l{index}-fifth",
                    onset,
                    min(6, next_onset - onset),
                    pitch + 7,
                    "lower",
                    articulations=articulation,
                )
            )
    return ScoreMaterial(
        material_id,
        40,
        tuple(notes),
        derived_from=f"a2-{name}",
        directions=_directions(material_id, 40, dynamic),
    )


def _b_material(
    name: str,
    center: int,
    lower_melody: tuple[int, ...],
    upper_tops: tuple[int, ...],
    dynamic: str,
    derived_from: str | None,
) -> ScoreMaterial:
    material_id = f"b-{name}"
    notes: list[ScoreNote] = []
    for index, (onset, interval) in enumerate(zip(B_LOWER_ONSETS, lower_melody, strict=True)):
        next_onset = B_LOWER_ONSETS[index + 1] if index + 1 < len(B_LOWER_ONSETS) else 34
        notes.append(
            ScoreNote(
                f"{material_id}-l{index}",
                onset,
                min(2, next_onset - onset),
                center - 17 + interval,
                "lower",
                articulations=("accent",) if index % 4 == 0 else ("staccato",),
            )
        )
    for index, (onset, interval) in enumerate(zip(B_UPPER_ONSETS, upper_tops, strict=True)):
        next_onset = B_UPPER_ONSETS[index + 1] if index + 1 < len(B_UPPER_ONSETS) else 34
        pitch = center + interval
        duration = min(6, next_onset - onset)
        for suffix, chord_pitch in (("top", pitch), ("third", _third_below(pitch))):
            notes.append(
                ScoreNote(
                    f"{material_id}-u{index}-{suffix}",
                    onset,
                    duration,
                    chord_pitch,
                    "upper",
                    articulations=("accent",),
                )
            )
    return ScoreMaterial(
        material_id,
        34,
        tuple(notes),
        derived_from=derived_from,
        directions=_directions(material_id, 34, dynamic),
    )


def _ending() -> ScoreMaterial:
    return ScoreMaterial(
        "ending",
        24,
        (
            ScoreNote("ending-u0", 0, 3, 69, "upper", articulations=("tenuto",)),
            ScoreNote("ending-l0", 0, 6, 45, "lower", articulations=("tenuto",)),
            ScoreNote("ending-l0-fifth", 0, 6, 52, "lower", articulations=("tenuto",)),
            ScoreNote("ending-u1", 3, 3, 67, "upper"),
            ScoreNote("ending-u2", 6, 3, 65, "upper"),
            ScoreNote("ending-l1", 6, 6, 48, "lower"),
            ScoreNote("ending-u3", 9, 3, 64, "upper"),
            ScoreNote("ending-u4", 12, 4, 62, "upper"),
            ScoreNote("ending-l2", 12, 8, 52, "lower"),
            ScoreNote("ending-u5", 16, 4, 64, "upper"),
            ScoreNote("ending-u6", 20, 4, 69, "upper", articulations=("tenuto",)),
            ScoreNote("ending-u6-third", 20, 4, 64, "upper", articulations=("tenuto",)),
            ScoreNote("ending-u6-fifth", 20, 4, 60, "upper", articulations=("tenuto",)),
            ScoreNote("ending-l3", 20, 4, 45, "lower", articulations=("tenuto",)),
            ScoreNote("ending-l3-octave", 20, 4, 57, "lower", articulations=("tenuto",)),
        ),
        directions=(
            ScoreDirection("ending-dynamic", 0, "dynamic", "p"),
            ScoreDirection("ending-breath", 20, "breath", "full"),
        ),
    )


def _score() -> ScoreSpec:
    aligned = tuple(_aligned_material(*spec) for spec in CORE_SPECS)
    broken = tuple(
        _broken_material(name, center, melody, bass, dynamic)
        for name, center, melody, bass, dynamic, _ in CORE_SPECS
    )
    b_specs = (
        ("theme", 72, (0, 3, 2, 7, 5, 3, 8, 7, 5, 2, 3, 0, -2), (0, 3, 7, 5, 0), "mf", None),
        (
            "theme-prime",
            72,
            (0, 3, 2, 7, 8, 5, 3, 7, 10, 8, 5, 3, 0),
            (0, 3, 7, 5, 3),
            "mf",
            "b-theme",
        ),
        ("contrast", 76, (0, -2, 0, 3, 7, 5, 8, 7, 3, 0, -2, 0, 3), (0, -2, 3, 0, -2), "f", None),
        (
            "contrast-prime",
            76,
            (0, -2, 0, 3, 5, 8, 10, 7, 5, 3, 2, 0, -2),
            (0, -2, 3, 0, 3),
            "f",
            "b-contrast",
        ),
        ("return", 72, (0, 3, 2, 7, 10, 8, 7, 3, 5, 2, -2, 0, 3), (0, 3, 7, 5, 0), "mf", "b-theme"),
        (
            "return-prime",
            72,
            (0, 3, 2, 7, 8, 5, 3, 7, 10, 8, 3, 0, -2),
            (0, 3, 7, 5, 3),
            "mp",
            "b-theme-prime",
        ),
    )
    b_materials = tuple(_b_material(*spec) for spec in b_specs)
    return ScoreSpec(
        "multiscale-score-v3",
        4,
        (*aligned, *broken, *b_materials, _ending()),
    )


def _plan() -> PiecePlan:
    nodes: list[PlanNode] = [PlanNode("root", None, 0, "whole")]
    sections = (
        ("a1", "opening", None, None, 9),
        ("b", "contrast", None, "a1", 0),
        ("a2", "return", "a1", None, 9),
    )
    materials = {
        "a1": (
            ("a1-theme", "a1-theme-prime"),
            ("a1-contrast", "a1-contrast-prime"),
            ("a1-return", "a1-return-prime"),
        ),
        "b": (
            ("b-theme", "b-theme-prime"),
            ("b-contrast", "b-contrast-prime"),
            ("b-return", "b-return-prime"),
        ),
        "a2": (
            ("a2-theme", "a2-theme-prime"),
            ("a2-contrast", "a2-contrast-prime"),
            ("a2-return", "a2-return-prime", "ending"),
        ),
    }
    for section_order, (
        section_id,
        role,
        section_source,
        contrast_source,
        focus,
    ) in enumerate(sections):
        nodes.append(
            PlanNode(
                section_id,
                "root",
                section_order,
                role,
                derived_from=section_source,
                contrasts_with=contrast_source,
            )
        )
        for phrase_order, phrase_role in enumerate(("statement", "contrast", "return")):
            phrase_id = f"{section_id}-{phrase_order}"
            if section_id == "a1":
                phrase_source = "a1-0" if phrase_role == "return" else None
            elif section_id == "b":
                phrase_source = "b-0" if phrase_role == "return" else None
            else:
                phrase_source = f"a1-{phrase_order}"
            nodes.append(
                PlanNode(
                    phrase_id,
                    section_id,
                    phrase_order,
                    phrase_role,
                    derived_from=phrase_source,
                )
            )
            for leaf_order, material_id in enumerate(materials[section_id][phrase_order]):
                leaf_id = f"{phrase_id}-{leaf_order}"
                if material_id == "ending":
                    leaf_source = None
                elif section_id == "a2":
                    leaf_source = f"a1-{phrase_order}-{leaf_order}"
                elif phrase_role == "return":
                    leaf_source = f"{section_id}-0-{leaf_order}"
                elif leaf_order == 1:
                    leaf_source = f"{phrase_id}-0"
                else:
                    leaf_source = None
                nodes.append(
                    PlanNode(
                        leaf_id,
                        phrase_id,
                        leaf_order,
                        "release"
                        if material_id == "ending"
                        else "variation"
                        if leaf_order
                        else "statement",
                        derived_from=leaf_source,
                        harmonic_focus=focus,
                        duration_weight=1,
                        score_material_id=material_id,
                    )
                )
    return PiecePlan(
        "multiscale-calibration-v3",
        "整形から崩しへ",
        9,
        "minor",
        "root",
        "tonic",
        tuple(nodes),
    )


def _performance() -> PerformanceSpec:
    items = [
        NodePerformance(
            "a1",
            timing_profile="savor",
            timing_amount="moderate",
            dynamics_profile="shape",
            articulation_profile="legato",
            coordination_profile="rolled",
        ),
        NodePerformance(
            "b",
            timing_profile="build",
            timing_amount="moderate",
            dynamics_profile="build",
            articulation_profile="score",
            coordination_profile="score",
        ),
        NodePerformance(
            "a2",
            timing_profile="flow",
            timing_amount="moderate",
            dynamics_profile="release",
            articulation_profile="light",
            coordination_profile="aligned",
        ),
    ]
    phrase_timings = {
        "a1": ("savor", "neutral", "release"),
        "b": ("build", "build", "release"),
        "a2": ("flow", "build", "release"),
    }
    for section in ("a1", "b", "a2"):
        for index, timing in enumerate(phrase_timings[section]):
            items.append(
                NodePerformance(
                    f"{section}-{index}",
                    timing_profile=timing,
                    timing_amount="subtle",
                    pedal_profile="clear" if section == "b" else "phrase_legato",
                )
            )
    return PerformanceSpec(
        "multiscale-performance-v3",
        180_000,
        62,
        "narrative-v1",
        tuple(items),
    )


def _write_inputs(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
) -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    input_dir = OUTPUT_DIR / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "plan": input_dir / "piece-plan.music.py",
        "score": input_dir / "score.music.py",
        "performance": input_dir / "performance.music.py",
    }
    paths["plan"].write_text(dump_piece_plan(plan) + "\n", encoding="utf-8")
    paths["score"].write_text(dump_score_spec(score) + "\n", encoding="utf-8")
    paths["performance"].write_text(
        dump_performance_spec(performance) + "\n",
        encoding="utf-8",
    )
    return (
        parse_piece_plan(paths["plan"].read_text(encoding="utf-8")),
        parse_score_spec(paths["score"].read_text(encoding="utf-8")),
        parse_performance_spec(paths["performance"].read_text(encoding="utf-8")),
    )


def main(maximum_target_ratio: float = 0.85) -> int:
    output_dir = OUTPUT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    plan, score, performance = _write_inputs(_plan(), _score(), _performance())
    validate_pipeline(plan, score, performance)
    rendered = render_performance(plan, score, performance)

    derived_targets = tuple(node.node_id for node in plan.nodes if node.derived_from is not None)
    recurrences = analyze_recurrences(
        plan,
        score,
        rendered,
        declared_cues={node_id: ("motif_head",) for node_id in derived_targets},
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
        maximum_target_ratio=maximum_target_ratio,
    )
    contrast = analyze_section_contrast(plan, score, target_node_id="b")
    material_alignments = tuple(
        analyze_material_vertical_alignment(item, minimum_ratio=0.60)
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
    leaf_alignments = tuple(
        analyze_vertical_alignment(
            plan,
            rendered,
            source_node_id=node.derived_from,
            target_node_id=node.node_id,
            minimum_target_ratio=0.60,
            minimum_increase=0.25,
        )
        for node in plan.nodes
        if node.node_id.startswith("a2-")
        and node.score_material_id is not None
        and node.score_material_id != "ending"
        and node.derived_from is not None
    )

    failures = [
        f"recurrence:{item.target_node_id}:{item.relation_status}"
        for item in recurrences
        if item.relation_status != "related" or item.exact_surface_copy
    ]
    failures.extend(
        f"motif:{item.target_node_id}"
        for item in recurrences
        if "motif_head" not in item.satisfied_identity_cues
    )
    failures.extend(
        f"voice-texture:{item.material_id}" for item in voice_textures if not item.passes
    )
    failures.extend(
        f"piano-texture:{item.material_id}" for item in piano_textures if not item.passes
    )
    failures.extend(f"breath:{item.node_id}" for item in breaths if not item.passes)
    failures.extend(
        f"score-alignment:{item.material_id}" for item in material_alignments if not item.passes
    )
    failures.extend(
        f"performance-alignment:{item.target_node_id}"
        for item in (section_alignment, *leaf_alignments)
        if not item.passes
    )
    if not pacing.passes:
        failures.append("pacing:a2")
    if not contrast.passes:
        failures.append("contrast:b")
    if failures:
        raise ValueError("calibration gates failed: " + ", ".join(failures))

    musicxml = render_musicxml(plan, score, output_dir / "score.musicxml")
    smf = render_performance_smf(rendered, output_dir / "final.mid")
    result = {
        "status": "passed",
        "duration_ms": smf.duration_ms,
        "note_count": smf.note_count,
        "musicxml": str(musicxml.resolve()),
        "smf": str(smf.path.resolve()),
        "lineage": rendered.lineage,
        "generation_order": {
            "canonical": [item.material_id for item in score.materials[:6]],
            "derived_broken": [item.material_id for item in score.materials[6:12]],
            "performance_form": ["a1", "b", "a2"],
        },
        "contrast": {
            "target": contrast.target_node_id,
            "source": contrast.source_node_id,
            "changed_axes": contrast.changed_axes,
            "shared_pitch_class_ratio": contrast.shared_pitch_class_ratio,
        },
        "material_alignments": [
            {
                "material_id": item.material_id,
                "shared_attack_ratio": item.shared_attack_ratio,
                "internal_independence": item.internal_independence,
            }
            for item in material_alignments
        ],
        "performance_alignments": [
            {
                "source": item.source_node_id,
                "target": item.target_node_id,
                "source_ratio": item.source_ratio,
                "target_ratio": item.target_ratio,
                "increase": item.target_ratio - item.source_ratio,
            }
            for item in (section_alignment, *leaf_alignments)
        ],
        "pacing": {
            "source_ms_per_unit": pacing.source_ms_per_unit,
            "target_ms_per_unit": pacing.target_ms_per_unit,
            "target_ratio": pacing.target_ms_per_unit / pacing.source_ms_per_unit,
            "source_note_count": pacing.source_note_count,
            "target_note_count": pacing.target_note_count,
        },
        "recurrences": [
            {
                "target": item.target_node_id,
                "source": item.source_node_id,
                "score_difference": item.score_difference,
                "performance_difference": item.performance_difference,
                "identity_cues": item.satisfied_identity_cues,
                "status": item.relation_status,
            }
            for item in recurrences
        ],
    }
    result_path = OUTPUT_DIR / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(result_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
