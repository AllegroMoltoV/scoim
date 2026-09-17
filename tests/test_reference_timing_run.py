from __future__ import annotations

import json
from pathlib import Path

import mido

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    performance_time_map,
    render_performance,
    render_performance_smf,
)
from llm_musical_composer.pipeline_dsl import (
    dump_performance_spec,
    dump_piece_plan,
    dump_score_spec,
)
from llm_musical_composer.reference_timing_run import (
    KnownTimingSource,
    run_known_timing_compatibility,
)


def _fixture() -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    plan = PiecePlan(
        "plan",
        "fixture",
        0,
        "major",
        "root",
        "tonic",
        (
            PlanNode("root", None, 0, "whole"),
            PlanNode(
                "leaf",
                "root",
                0,
                "statement",
                duration_weight=1,
                score_material_id="material",
            ),
        ),
    )
    score = ScoreSpec(
        "score",
        4,
        (
            ScoreMaterial(
                "material",
                4,
                (
                    ScoreNote("low", 0, 1, 48, "lower"),
                    ScoreNote("high", 0, 1, 60, "upper"),
                    ScoreNote("middle", 1, 1, 62, "upper"),
                    ScoreNote("ending", 3, 1, 64, "upper"),
                ),
            ),
        ),
    )
    performance = PerformanceSpec(
        "performance",
        4_000,
        64,
        "subtle-v1",
        (NodePerformance("root", coordination_profile="rolled", pedal_profile="phrase_legato"),),
    )
    return plan, score, performance


def _source(root: Path) -> KnownTimingSource:
    plan, score, performance = _fixture()
    inputs = root / "inputs"
    outputs = root / "outputs"
    inputs.mkdir(parents=True)
    outputs.mkdir()
    piece_path = inputs / "piece.music.py"
    score_path = inputs / "score.music.py"
    performance_path = inputs / "performance.music.py"
    piece_path.write_text(dump_piece_plan(plan), encoding="utf-8")
    score_path.write_text(dump_score_spec(score), encoding="utf-8")
    performance_path.write_text(dump_performance_spec(performance), encoding="utf-8")
    smf_path = outputs / "final.mid"
    rendered = render_performance(plan, score, performance)
    render_performance_smf(rendered, smf_path)
    evidence_path = root / "result.json"
    evidence_path.write_text(
        json.dumps({"status": "passed", "lineage": list(rendered.lineage)}),
        encoding="utf-8",
    )
    return KnownTimingSource(
        case_id="fixture-v1",
        group_id="fixture-family",
        root=root,
        piece_path=piece_path,
        score_path=score_path,
        performance_path=performance_path,
        smf_path=smf_path,
        evidence_kind="calibration_result",
        evidence_path=evidence_path,
    )


def test_known_source_compatibility_rerenders_and_records_lineage(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    plan, score, performance = _fixture()
    time_map = performance_time_map(plan, score, performance)
    assert time_map[0] == 0
    assert time_map[-1] == performance.target_duration_ms
    assert len(time_map) == 5

    result = run_known_timing_compatibility(sources=(source,), output_dir=tmp_path / "output")

    assert result["status"] == "pass"
    assert result["source_count"] == 1
    record = json.loads((tmp_path / "output" / "known-source-availability.jsonl").read_text())
    assert record["status"] == "affirmative_evidence"
    assert record["event_roundtrip_status"] == "pass"
    assert record["lineage_status"] == "pass"
    assert record["timing_roundtrip_status"] == "pass"
    assert record["feature_roundtrip_status"]["cc64"] == "pass"
    assert record["group_id"] == "fixture-family"
    manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
    assert manifest["status"] == "pass"
    assert set(manifest["outputs"]) >= {
        "finite-vocabulary-collisions.json",
        "known-score-results.jsonl",
        "known-source-availability.jsonl",
        "non-identifiability-controls.json",
        "result.json",
        "run-spec.json",
        "sequential-stage-assessment.json",
    }
    known_score = json.loads((tmp_path / "output" / "known-score-results.jsonl").read_text())
    assert known_score["status"] == "assessed"
    assert known_score["unmatched_expected_count"] == 0
    assert known_score["unmatched_observed_count"] == 0
    assert record["score_timing_candidate_recovery"]["status"] == "recovered"
    assert record["score_timing_candidate_recovery"]["candidate_count"] == 27
    assert record["score_timing_candidate_recovery"]["equivalent_candidate_count"] > 0
    grouping_diagnostics = record["score_timing_candidate_recovery"]["grouping_diagnostics"]
    assert grouping_diagnostics["attack-30ms"]["status"] == ("splits_known_score_positions")
    assert grouping_diagnostics["rolled-merge-60ms"]["status"] == "compatible"
    assert (
        record["score_timing_candidate_recovery"]["time_map_oracle_diagnostics"]["status"]
        == "assessed"
    )
    assert result["score_timing_candidate_recovered_count"] == 1
    assessment = json.loads((tmp_path / "output" / "sequential-stage-assessment.json").read_text())
    assert assessment["known_score_note_on_assessed_count"] == 1
    assert assessment["recommendation"] == "coupled_score_timing_hypothesis"


def test_known_source_lineage_mismatch_is_not_reported_as_success(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    source.evidence_path.write_text(
        json.dumps({"status": "passed", "lineage": ["wrong"]}), encoding="utf-8"
    )

    result = run_known_timing_compatibility(sources=(source,), output_dir=tmp_path / "output")

    assert result["status"] == "partial"
    record = json.loads((tmp_path / "output" / "known-source-availability.jsonl").read_text())
    assert record["status"] == "investigated_no_evidence"
    assert record["lineage_status"] == "fail"


def test_known_source_separates_note_timing_from_pedal_difference(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    midi = mido.MidiFile(source.smf_path)
    track = midi.tracks[1]
    for index, message in enumerate(track[:-1]):
        if (
            message.type == "control_change"
            and message.control == 64
            and message.time >= 10
            and track[index + 1].time >= 10
        ):
            message.time += 10
            track[index + 1].time -= 10
            break
    else:
        raise AssertionError("movable CC64 fixture event was not found")
    midi.save(source.smf_path)

    result = run_known_timing_compatibility(sources=(source,), output_dir=tmp_path / "output")

    record = json.loads((tmp_path / "output" / "known-source-availability.jsonl").read_text())
    assert result["status"] == "partial"
    assert record["event_roundtrip_status"] == "fail"
    assert record["timing_roundtrip_status"] == "pass"
    assert record["feature_roundtrip_status"] == {
        "cc64": "fail",
        "note_off": "pass",
        "note_on": "pass",
    }
