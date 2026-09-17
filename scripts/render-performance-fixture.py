"""多段IRの最小fixtureをMusicXMLと二種類の演奏SMFへ書き出す。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
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
)


def _fixture() -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    plan = PiecePlan(
        plan_id="two-renditions",
        title="同じ主題の異なる演奏",
        tonal_center=9,
        mode="minor",
        root_node_id="root",
        ending_intent="tonic",
        nodes=(
            PlanNode("root", None, 0, "whole"),
            PlanNode("a1", "root", 0, "opening"),
            PlanNode(
                "a1-leaf",
                "a1",
                0,
                "statement",
                harmonic_focus=9,
                duration_weight=1,
                score_material_id="theme",
            ),
            PlanNode(
                "bridge",
                "root",
                1,
                "contrast",
                harmonic_focus=0,
                duration_weight=1,
                score_material_id="bridge-material",
            ),
            PlanNode("a2", "root", 2, "return", derived_from="a1"),
            PlanNode(
                "a2-leaf",
                "a2",
                0,
                "return",
                derived_from="a1-leaf",
                harmonic_focus=9,
                duration_weight=1,
                score_material_id="theme",
            ),
        ),
    )
    score = ScoreSpec(
        score_id="score",
        divisions=4,
        materials=(
            ScoreMaterial(
                material_id="theme",
                length_units=8,
                notes=(
                    ScoreNote("theme-u1", 0, 4, 64, "upper", articulations=("tenuto",)),
                    ScoreNote("theme-l1", 0, 4, 48, "lower"),
                    ScoreNote("theme-u2", 4, 2, 67, "upper", articulations=("staccato",)),
                    ScoreNote("theme-l2", 4, 4, 52, "lower"),
                ),
                directions=(
                    ScoreDirection("theme-dyn", 0, "dynamic", "mf"),
                    ScoreDirection("theme-breath", 6, "breath", "light"),
                ),
            ),
            ScoreMaterial(
                material_id="bridge-material",
                length_units=4,
                notes=(
                    ScoreNote("bridge-u", 0, 4, 69, "upper", articulations=("accent",)),
                    ScoreNote("bridge-l", 0, 4, 45, "lower"),
                ),
                directions=(ScoreDirection("bridge-dyn", 0, "dynamic", "f"),),
            ),
        ),
    )
    performance = PerformanceSpec(
        performance_id="shaped",
        target_duration_ms=12_000,
        default_velocity=64,
        timing_budget_id="subtle-v1",
        node_performances=(
            NodePerformance(
                "a1",
                timing_profile="savor",
                timing_amount="moderate",
                dynamics_profile="shape",
                articulation_profile="legato",
                pedal_profile="phrase_legato",
            ),
            NodePerformance(
                "bridge",
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
                dynamics_profile="steady",
                articulation_profile="light",
                pedal_profile="phrase_legato",
            ),
        ),
    )
    return plan, score, performance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".appendix/performance-pipeline-minimal"),
    )
    args = parser.parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plan, score, shaped = _fixture()
    flat = replace(
        shaped,
        performance_id="flat",
        node_performances=tuple(
            replace(
                item,
                timing_profile="neutral",
                dynamics_profile="steady",
                articulation_profile="score",
            )
            for item in shaped.node_performances
        ),
    )
    score_path = render_musicxml(plan, score, output_dir / "score.musicxml")
    shaped_result = render_performance(plan, score, shaped)
    flat_result = render_performance(plan, score, flat)
    shaped_smf = render_performance_smf(shaped_result, output_dir / "shaped.mid")
    flat_smf = render_performance_smf(flat_result, output_dir / "flat.mid")
    summary = {
        "score": str(score_path.resolve()),
        "shaped_smf": str(shaped_smf.path.resolve()),
        "flat_smf": str(flat_smf.path.resolve()),
        "duration_ms": shaped_smf.duration_ms,
        "note_count": shaped_smf.note_count,
        "same_score_identity": [(note.pitch, note.voice) for note in shaped_result.notes]
        == [(note.pitch, note.voice) for note in flat_result.notes],
        "performance_differs": [
            (note.at_ms, note.duration_ms, note.velocity) for note in shaped_result.notes
        ]
        != [(note.at_ms, note.duration_ms, note.velocity) for note in flat_result.notes],
        "lineage": shaped_result.lineage,
    }
    summary_path = output_dir / "result.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
