"""初回試聴の局所解を修正した3分校正曲を決定的に生成する。"""

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
    analyze_material_voice_texture,
    analyze_piano_texture_variety,
    analyze_section_pacing,
)

OUTPUT_DIR = Path(".appendix/multiscale-calibration-run-v2")
A_ONSETS = (0, 4, 7, 11, 15, 18, 22, 26, 30, 34)
B_ONSETS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 34)
A2_ONSETS = (0, 2, 4, 6, 9, 12, 15, 18, 21, 24, 27, 30, 32, 34)
A_BASS_ONSETS = (0, 6, 12, 18, 24, 30, 34)
B_BASS_ONSETS = (0, 4, 8, 12, 16, 20, 24, 28, 34)
A2_BASS_ONSETS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 34)
A_BASS = (0, 7, 3, 8, 5, 7, 0)
B_BASS = (0, 7, 3, 10, 5, 8, 7, 3, 0)
A2_BASS = (0, 7, 12, 7, 3, 10, 5, 8, 7, 0)


def _third_below(pitch: int) -> int:
    a_minor = {9, 11, 0, 2, 4, 5, 7}
    for distance in (3, 4):
        candidate = pitch - distance
        if candidate % 12 in a_minor:
            return candidate
    return pitch - 3


def _material(
    material_id: str,
    *,
    center: int,
    melody: tuple[int, ...],
    upper_onsets: tuple[int, ...],
    bass: tuple[int, ...],
    bass_onsets: tuple[int, ...],
    upper_dyads: frozenset[int],
    lower_dyads: frozenset[int],
    character: str,
    dynamic: str,
    derived_from: str | None = None,
) -> ScoreMaterial:
    if len(melody) != len(upper_onsets) or len(bass) != len(bass_onsets):
        raise ValueError("material blueprint lengths must match")
    notes: list[ScoreNote] = []
    for index, (onset, interval) in enumerate(zip(upper_onsets, melody, strict=True)):
        next_onset = upper_onsets[index + 1] if index + 1 < len(upper_onsets) else 40
        duration = min(4, next_onset - onset)
        pitch = center + interval
        articulations: tuple[str, ...] = ()
        if character == "spacious" and index in {0, len(melody) - 1}:
            articulations = ("tenuto",)
        elif character == "accented" and onset in upper_dyads:
            articulations = ("accent",)
        elif character == "brisk" and index % 4 == 2:
            articulations = ("staccato",)
        notes.append(
            ScoreNote(
                f"{material_id}-u{index}",
                onset,
                duration,
                pitch,
                "upper",
                articulations=articulations,
            )
        )
        if onset in upper_dyads:
            notes.append(
                ScoreNote(
                    f"{material_id}-u{index}-third",
                    onset,
                    duration,
                    _third_below(pitch),
                    "upper",
                    articulations=articulations,
                )
            )
    bass_root = center - 24
    for index, (onset, interval) in enumerate(zip(bass_onsets, bass, strict=True)):
        next_onset = bass_onsets[index + 1] if index + 1 < len(bass_onsets) else 40
        duration = min(6, next_onset - onset)
        pitch = bass_root + interval
        notes.append(
            ScoreNote(
                f"{material_id}-l{index}",
                onset,
                duration,
                pitch,
                "lower",
                articulations=("tenuto",) if onset in {0, 34} else (),
            )
        )
        if onset in lower_dyads:
            notes.append(
                ScoreNote(
                    f"{material_id}-l{index}-fifth",
                    onset,
                    duration,
                    pitch + 7,
                    "lower",
                    articulations=("tenuto",) if onset in {0, 34} else (),
                )
            )
    return ScoreMaterial(
        material_id=material_id,
        length_units=40,
        notes=tuple(notes),
        derived_from=derived_from,
        directions=(
            ScoreDirection(f"{material_id}-dynamic", 0, "dynamic", dynamic),
            ScoreDirection(f"{material_id}-breath", 36, "breath", "light"),
        ),
    )


def _ending() -> ScoreMaterial:
    return ScoreMaterial(
        material_id="ending",
        length_units=24,
        notes=(
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
    spacious = {
        "upper_onsets": A_ONSETS,
        "bass": A_BASS,
        "bass_onsets": A_BASS_ONSETS,
        "upper_dyads": frozenset({11, 26, 34}),
        "lower_dyads": frozenset({0, 18, 34}),
        "character": "spacious",
    }
    accented = {
        "upper_onsets": B_ONSETS,
        "bass": B_BASS,
        "bass_onsets": B_BASS_ONSETS,
        "upper_dyads": frozenset({0, 9, 18, 27, 34}),
        "lower_dyads": frozenset({0, 12, 24, 34}),
        "character": "accented",
    }
    brisk = {
        "upper_onsets": A2_ONSETS,
        "bass": A2_BASS,
        "bass_onsets": A2_BASS_ONSETS,
        "upper_dyads": frozenset({0, 12, 24, 34}),
        "lower_dyads": frozenset({0, 16, 34}),
        "character": "brisk",
    }
    definitions = (
        _material(
            "a1-theme",
            center=69,
            melody=(0, 2, 3, 7, 5, 3, 2, -2, 0, 3),
            dynamic="mp",
            **spacious,
        ),
        _material(
            "a1-theme-prime",
            center=69,
            melody=(0, 2, 3, 7, 8, 7, 5, 3, 2, 0),
            dynamic="mp",
            derived_from="a1-theme",
            **spacious,
        ),
        _material(
            "a1-contrast",
            center=72,
            melody=(0, 3, 7, 5, 8, 7, 3, 5, 2, 0),
            dynamic="mf",
            **spacious,
        ),
        _material(
            "a1-contrast-prime",
            center=72,
            melody=(0, 3, 7, 5, 3, 0, -2, 2, 5, 3),
            dynamic="mf",
            derived_from="a1-contrast",
            **spacious,
        ),
        _material(
            "a1-return",
            center=69,
            melody=(0, 2, 3, 7, 3, 5, 8, 7, 2, 0),
            dynamic="mp",
            derived_from="a1-theme",
            **spacious,
        ),
        _material(
            "a1-return-prime",
            center=69,
            melody=(0, 2, 3, 7, 10, 8, 5, 3, -2, 0),
            dynamic="p",
            derived_from="a1-theme-prime",
            **spacious,
        ),
        _material(
            "b-theme",
            center=72,
            melody=(0, 3, 7, 5, 8, 10, 8, 7, 3, 5, 2, 0),
            dynamic="mf",
            **accented,
        ),
        _material(
            "b-theme-prime",
            center=72,
            melody=(0, 3, 7, 5, 3, 7, 10, 8, 5, 2, 3, 0),
            dynamic="mf",
            derived_from="b-theme",
            **accented,
        ),
        _material(
            "b-contrast",
            center=76,
            melody=(0, -2, 0, 3, 7, 5, 8, 7, 3, 0, -2, 0),
            dynamic="f",
            **accented,
        ),
        _material(
            "b-contrast-prime",
            center=76,
            melody=(0, -2, 0, 3, 5, 8, 10, 7, 5, 3, 2, 0),
            dynamic="f",
            derived_from="b-contrast",
            **accented,
        ),
        _material(
            "b-return",
            center=72,
            melody=(0, 3, 7, 5, 10, 8, 7, 3, 5, 2, -2, 0),
            dynamic="mf",
            derived_from="b-theme",
            **accented,
        ),
        _material(
            "b-return-prime",
            center=72,
            melody=(0, 3, 7, 5, 8, 5, 3, 7, 10, 8, 3, 0),
            dynamic="mp",
            derived_from="b-theme-prime",
            **accented,
        ),
        _material(
            "a2-theme",
            center=69,
            melody=(0, 2, 3, 7, 10, 8, 5, 7, 3, 2, 5, 3, -2, 0),
            dynamic="mp",
            derived_from="a1-theme",
            **brisk,
        ),
        _material(
            "a2-theme-prime",
            center=69,
            melody=(0, 2, 3, 7, 5, 8, 12, 10, 7, 5, 3, 0, 2, 0),
            dynamic="mp",
            derived_from="a1-theme-prime",
            **brisk,
        ),
        _material(
            "a2-contrast",
            center=72,
            melody=(0, 3, 7, 5, 10, 8, 7, 3, 5, 8, 3, 2, -2, 0),
            dynamic="mf",
            derived_from="a1-contrast",
            **brisk,
        ),
        _material(
            "a2-contrast-prime",
            center=72,
            melody=(0, 3, 7, 5, 3, 0, 5, 8, 10, 7, 5, 3, 2, 0),
            dynamic="mf",
            derived_from="a1-contrast-prime",
            **brisk,
        ),
        _material(
            "a2-return",
            center=69,
            melody=(0, 2, 3, 7, 12, 10, 8, 5, 7, 3, 5, 2, -2, 0),
            dynamic="mp",
            derived_from="a1-return",
            **brisk,
        ),
        _material(
            "a2-return-prime",
            center=69,
            melody=(0, 2, 3, 7, 8, 12, 10, 7, 5, 3, 0, 2, -2, 0),
            dynamic="p",
            derived_from="a1-return-prime",
            **brisk,
        ),
        _ending(),
    )
    return ScoreSpec("multiscale-score-v2", 4, definitions)


def _plan() -> PiecePlan:
    nodes: list[PlanNode] = [PlanNode("root", None, 0, "whole")]
    sections = (
        ("a1", "opening", None, 9),
        ("b", "contrast", None, 0),
        ("a2", "return", "a1", 9),
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
    phrase_roles = ("statement", "contrast", "return")
    for section_order, (section_id, role, section_source, focus) in enumerate(sections):
        nodes.append(
            PlanNode(
                section_id,
                "root",
                section_order,
                role,
                derived_from=section_source,
            )
        )
        for phrase_order, phrase_role in enumerate(phrase_roles):
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
                leaf_role = (
                    "release"
                    if material_id == "ending"
                    else "variation"
                    if leaf_order
                    else "statement"
                )
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
                        leaf_role,
                        derived_from=leaf_source,
                        harmonic_focus=focus,
                        duration_weight=1,
                        score_material_id=material_id,
                    )
                )
    return PiecePlan(
        "multiscale-calibration-v2",
        "多尺度の帰還 第二稿",
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
        ),
        NodePerformance(
            "b",
            timing_profile="build",
            timing_amount="moderate",
            dynamics_profile="build",
            articulation_profile="score",
        ),
        NodePerformance(
            "a2",
            timing_profile="flow",
            timing_amount="moderate",
            dynamics_profile="release",
            articulation_profile="light",
        ),
    ]
    for section in ("a1", "b", "a2"):
        for index, timing in enumerate(("neutral", "build", "release")):
            items.append(
                NodePerformance(
                    f"{section}-{index}",
                    timing_profile=timing,
                    timing_amount="subtle",
                    pedal_profile="phrase_legato" if section != "b" else "clear",
                )
            )
    return PerformanceSpec(
        "multiscale-performance-v2",
        180_000,
        62,
        "narrative-v1",
        tuple(items),
    )


def main() -> int:
    input_dir = OUTPUT_DIR / "inputs"
    output_dir = OUTPUT_DIR / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_plan, original_score, original_performance = _plan(), _score(), _performance()
    paths = {
        "plan": input_dir / "piece-plan.music.py",
        "score": input_dir / "score.music.py",
        "performance": input_dir / "performance.music.py",
    }
    paths["plan"].write_text(dump_piece_plan(original_plan) + "\n", encoding="utf-8")
    paths["score"].write_text(dump_score_spec(original_score) + "\n", encoding="utf-8")
    paths["performance"].write_text(
        dump_performance_spec(original_performance) + "\n",
        encoding="utf-8",
    )
    plan = parse_piece_plan(paths["plan"].read_text(encoding="utf-8"))
    score = parse_score_spec(paths["score"].read_text(encoding="utf-8"))
    performance = parse_performance_spec(paths["performance"].read_text(encoding="utf-8"))
    validate_pipeline(plan, score, performance)
    rendered = render_performance(plan, score, performance)
    derived_targets = tuple(node.node_id for node in plan.nodes if node.derived_from is not None)
    assessments = analyze_recurrences(
        plan,
        score,
        rendered,
        declared_cues={node_id: ("motif_head", "pitch_contour") for node_id in derived_targets},
    )
    voice_textures = tuple(analyze_material_voice_texture(material) for material in score.materials)
    piano_textures = tuple(analyze_piano_texture_variety(material) for material in score.materials)
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
        maximum_target_ratio=0.85,
    )
    failures = [
        f"recurrence:{item.target_node_id}:{item.relation_status}"
        for item in assessments
        if item.relation_status != "related" or item.exact_surface_copy
    ]
    failures.extend(
        f"motif:{item.target_node_id}"
        for item in assessments
        if "motif_head" not in item.satisfied_identity_cues
    )
    failures.extend(
        f"voice-texture:{item.material_id}" for item in voice_textures if not item.passes
    )
    failures.extend(
        f"piano-texture:{item.material_id}" for item in piano_textures if not item.passes
    )
    failures.extend(f"breath:{item.node_id}" for item in breaths if not item.passes)
    if not pacing.passes:
        failures.append("pacing:a2")
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
        "recurrences": [
            {
                "target": item.target_node_id,
                "source": item.source_node_id,
                "score_difference": item.score_difference,
                "performance_difference": item.performance_difference,
                "identity_cues": item.satisfied_identity_cues,
                "status": item.relation_status,
            }
            for item in assessments
        ],
        "piano_textures": [
            {
                "material_id": item.material_id,
                "attack_size_counts": item.attack_size_counts,
                "maximum_polyphony": item.maximum_polyphony,
            }
            for item in piano_textures
        ],
        "boundary_breaths": [
            {
                "node_id": item.node_id,
                "new_attack_gap_units": item.new_attack_gap_units,
            }
            for item in breaths
        ],
        "pacing": {
            "source_node_id": pacing.source_node_id,
            "target_node_id": pacing.target_node_id,
            "source_ms_per_unit": pacing.source_ms_per_unit,
            "target_ms_per_unit": pacing.target_ms_per_unit,
            "target_ratio": pacing.target_ms_per_unit / pacing.source_ms_per_unit,
            "source_note_count": pacing.source_note_count,
            "target_note_count": pacing.target_note_count,
        },
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
