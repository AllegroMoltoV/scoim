"""多尺度の変奏付き再現を持つ3分校正曲を決定的に生成する。"""

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
)

OUTPUT_DIR = Path(".appendix/multiscale-calibration-run")
UPPER_ONSETS = (0, 3, 7, 10, 14, 18, 21, 25, 28, 32)
LOWER_ONSETS = (0, 8, 16, 24, 32)
BASE_CONTOUR = (0, 2, 3, 7, 5, 3, 2, -2, 0, 0)
CONTRAST_CONTOUR = (0, 4, 2, 5, 7, 9, 7, 4, 2, 0)


def _material(
    material_id: str,
    *,
    center: int,
    contour: tuple[int, ...],
    derived_from: str | None = None,
    rhythmic_variant: bool = False,
    dynamic: str = "mp",
    final_chord: bool = False,
) -> ScoreMaterial:
    notes: list[ScoreNote] = []
    for index, (onset, interval) in enumerate(zip(UPPER_ONSETS, contour, strict=True)):
        adjusted_onset = onset + (1 if rhythmic_variant and index in {1, 3, 6, 8} else 0)
        duration = 2 if index % 3 else 3
        if index == len(UPPER_ONSETS) - 1:
            duration = 8
        articulations = ("tenuto",) if index in {0, len(UPPER_ONSETS) - 1} else ()
        if index in {3, 7}:
            articulations = ("staccato",)
        notes.append(
            ScoreNote(
                f"{material_id}-u{index}",
                adjusted_onset,
                duration,
                center + interval,
                "upper",
                articulations=articulations,
            )
        )
    bass_root = center - 24
    bass_pattern = (0, 7, 3, 5, 0)
    for index, (onset, interval) in enumerate(zip(LOWER_ONSETS, bass_pattern, strict=True)):
        notes.append(
            ScoreNote(
                f"{material_id}-l{index}",
                onset,
                7 if onset < 32 else 8,
                bass_root + interval,
                "lower",
                articulations=("tenuto",) if onset in {0, 32} else (),
            )
        )
    if final_chord:
        notes.extend(
            (
                ScoreNote(f"{material_id}-cad3", 32, 8, 72, "upper", articulations=("tenuto",)),
                ScoreNote(f"{material_id}-cad5", 32, 8, 76, "upper", articulations=("tenuto",)),
                ScoreNote(f"{material_id}-cadb", 32, 8, 57, "lower", articulations=("tenuto",)),
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


def _score() -> ScoreSpec:
    definitions = (
        _material("a1-theme", center=69, contour=BASE_CONTOUR),
        _material(
            "a1-theme-prime",
            center=69,
            contour=BASE_CONTOUR,
            derived_from="a1-theme",
            rhythmic_variant=True,
        ),
        _material("a1-contrast", center=72, contour=CONTRAST_CONTOUR, dynamic="mf"),
        _material(
            "a1-contrast-prime",
            center=72,
            contour=CONTRAST_CONTOUR,
            derived_from="a1-contrast",
            rhythmic_variant=True,
            dynamic="mf",
        ),
        _material(
            "a1-return",
            center=69,
            contour=BASE_CONTOUR,
            derived_from="a1-theme",
        ),
        _material(
            "a1-return-prime",
            center=69,
            contour=BASE_CONTOUR,
            derived_from="a1-theme-prime",
            rhythmic_variant=True,
        ),
        _material("b-theme", center=72, contour=CONTRAST_CONTOUR, dynamic="mf"),
        _material(
            "b-theme-prime",
            center=72,
            contour=CONTRAST_CONTOUR,
            derived_from="b-theme",
            rhythmic_variant=True,
            dynamic="mf",
        ),
        _material("b-contrast", center=76, contour=BASE_CONTOUR, dynamic="f"),
        _material(
            "b-contrast-prime",
            center=76,
            contour=BASE_CONTOUR,
            derived_from="b-contrast",
            rhythmic_variant=True,
            dynamic="f",
        ),
        _material(
            "b-return",
            center=72,
            contour=CONTRAST_CONTOUR,
            derived_from="b-theme",
            dynamic="mf",
        ),
        _material(
            "b-return-prime",
            center=72,
            contour=CONTRAST_CONTOUR,
            derived_from="b-theme-prime",
            rhythmic_variant=True,
            dynamic="mf",
        ),
        _material(
            "a2-theme-prime",
            center=69,
            contour=BASE_CONTOUR,
            derived_from="a1-theme-prime",
            rhythmic_variant=False,
            dynamic="p",
        ),
        _material(
            "a2-contrast",
            center=71,
            contour=CONTRAST_CONTOUR,
            derived_from="a1-contrast",
            dynamic="mp",
        ),
        _material(
            "a2-contrast-prime",
            center=71,
            contour=CONTRAST_CONTOUR,
            derived_from="a1-contrast-prime",
            rhythmic_variant=False,
            dynamic="mp",
        ),
        _material(
            "a2-return",
            center=69,
            contour=BASE_CONTOUR,
            derived_from="a1-return",
            rhythmic_variant=True,
            dynamic="p",
        ),
        _material(
            "a2-return-prime",
            center=69,
            contour=BASE_CONTOUR,
            derived_from="a1-return-prime",
            dynamic="p",
        ),
        _material(
            "ending",
            center=69,
            contour=BASE_CONTOUR,
            dynamic="p",
            final_chord=True,
        ),
    )
    return ScoreSpec("multiscale-score", 4, definitions)


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
            ("a1-theme", "a2-theme-prime"),
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
                        (
                            "release"
                            if material_id == "ending"
                            else "variation"
                            if leaf_order
                            else "statement"
                        ),
                        derived_from=leaf_source,
                        harmonic_focus=focus,
                        duration_weight=1,
                        score_material_id=material_id,
                    )
                )
    return PiecePlan(
        "multiscale-calibration",
        "多尺度の帰還",
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
            pedal_profile="phrase_legato",
        ),
        NodePerformance(
            "b",
            timing_profile="build",
            timing_amount="subtle",
            dynamics_profile="build",
            articulation_profile="score",
            pedal_profile="clear",
        ),
        NodePerformance(
            "a2",
            timing_profile="flow",
            timing_amount="subtle",
            dynamics_profile="release",
            articulation_profile="light",
            pedal_profile="phrase_legato",
        ),
    ]
    for section in ("a1", "b", "a2"):
        for index, timing in enumerate(("neutral", "build", "release")):
            items.append(
                NodePerformance(
                    f"{section}-{index}",
                    timing_profile=timing,
                    timing_amount="subtle",
                )
            )
    return PerformanceSpec(
        "multiscale-performance",
        180_000,
        62,
        "subtle-v1",
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
        declared_cues={node_id: ("pitch_contour",) for node_id in derived_targets},
    )
    textures = tuple(analyze_material_voice_texture(material) for material in score.materials)
    breaths = tuple(
        analyze_boundary_breath(plan, score, node_id, minimum_gap_units=4)
        for node_id in ("a1", "b")
    )
    failures = [
        f"recurrence:{item.target_node_id}:{item.relation_status}"
        for item in assessments
        if item.relation_status != "related" or item.exact_surface_copy
    ]
    failures.extend(f"texture:{item.material_id}" for item in textures if not item.passes)
    failures.extend(f"breath:{item.node_id}" for item in breaths if not item.passes)
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
        "voice_textures_passed": len(textures),
        "boundary_breaths": [
            {
                "node_id": item.node_id,
                "new_attack_gap_units": item.new_attack_gap_units,
            }
            for item in breaths
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
