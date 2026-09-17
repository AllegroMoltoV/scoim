from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mido
import pytest

from llm_musical_composer.harmony_observation_diagnostics import (
    HARMONY_QUALITIES,
    ObservationDiagnosticSource,
    build_group_support,
    harmony_keys,
    harmony_support,
    one_or_more_support_ids,
    pitch_class_inclusion_table,
    run_known_harmony_oracle,
    run_observation_diagnostics,
    stepwise_motion_evidence,
    verify_roundtrip_sources,
)
from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    render_performance,
    render_performance_smf,
)
from llm_musical_composer.pipeline_dsl import (
    dump_performance_spec,
    dump_piece_plan,
    dump_score_spec,
)
from llm_musical_composer.reference_decomposition import ObservedNote
from llm_musical_composer.reference_timing_run import KnownTimingSource
from llm_musical_composer.score_timing_hypothesis import GroupingAttack


def _note(
    event_id: str,
    *,
    pitch: int,
    onset_us: int,
    offset_us: int,
) -> ObservedNote:
    return ObservedNote(
        note_on_event_id=event_id,
        note_off_event_id=f"{event_id}-off",
        track=0,
        channel=0,
        pitch=pitch,
        velocity=64,
        note_off_velocity=0,
        onset_tick=onset_us // 1_000,
        offset_tick=offset_us // 1_000,
        onset_us=onset_us,
        offset_us=offset_us,
        matching_status="matched",
    )


def test_harmony_vocabulary_order_is_fixed() -> None:
    keys = harmony_keys()

    assert HARMONY_QUALITIES == ("major", "minor", "diminished", "major-seventh")
    assert len(keys) == 48
    assert keys[:4] == tuple((0, quality) for quality in HARMONY_QUALITIES)
    assert keys[-1] == (11, "major-seventh")
    assert pitch_class_inclusion_table()[0] == {
        "harmony_index": 0,
        "root_pitch_class": 0,
        "quality": "major",
        "pitch_classes": [0, 4, 7],
    }


def test_one_or_more_support_set_cannot_shrink_when_observed_pitches_are_added() -> None:
    one_pitch = one_or_more_support_ids({0})
    two_pitches = one_or_more_support_ids({0, 1})

    assert one_pitch < two_pitches
    assert one_pitch.issubset(two_pitches)


def test_attack_and_key_held_support_remain_separate() -> None:
    notes = (
        _note("c", pitch=48, onset_us=0, offset_us=500_000),
        _note("e", pitch=52, onset_us=0, offset_us=500_000),
        _note("g", pitch=55, onset_us=0, offset_us=500_000),
        _note("d", pitch=62, onset_us=100_000, offset_us=180_000),
        _note("f", pitch=65, onset_us=200_000, offset_us=280_000),
        _note("a", pitch=69, onset_us=300_000, offset_us=380_000),
    )
    notes_by_id = {note.note_on_event_id: note for note in notes}
    groups = tuple(
        GroupingAttack(at, (event_id,), (0,))
        for at, event_id in ((100_000, "d"), (200_000, "f"), (300_000, "a"))
    )

    result = build_group_support(
        groups=groups,
        notes_by_id=notes_by_id,
        note_matching_status="assessed",
    )

    middle = result[1]
    c_major = middle["harmony_support"][0]
    assert c_major["attack_support"]["chord_tone_count"] == 0
    assert c_major["key_held_support"]["chord_tone_count"] == 3
    assert c_major["key_held_support"]["pitch_count"] == 4
    assert "pedal_sounding_support" not in c_major


def test_ambiguous_matching_only_disables_key_held_support() -> None:
    note = _note("c", pitch=60, onset_us=0, offset_us=100_000)
    groups = (GroupingAttack(0, ("c",), (0,)),)

    result = build_group_support(
        groups=groups,
        notes_by_id={"c": note},
        note_matching_status="ambiguous",
    )

    record = result[0]["harmony_support"][0]
    assert record["attack_support"]["status"] == "assessed"
    assert record["key_held_support"] == {
        "status": "unable_to_investigate",
        "reason": "note_matching_not_assessed",
    }


def test_major_seventh_and_low_register_evidence_are_explicit() -> None:
    result = harmony_support((36, 40, 47, 48), root_pitch_class=0, quality="major-seventh")

    assert result["harmony_pitch_class_count"] == 4
    assert result["chord_tone_count"] == 4
    assert result["chord_tone_rate"] == {"numerator": 1, "denominator": 1}
    assert result["low_register"] == {
        "root_or_fifth_count": 1,
        "third_count": 1,
        "other_count": 1,
    }


def test_stepwise_pitch_shape_is_not_named_as_a_non_chord_tone_type() -> None:
    assert stepwise_motion_evidence((60, 62, 64)) == "monotonic_step"
    assert stepwise_motion_evidence((60, 61, 60)) == "returning_step"
    assert stepwise_motion_evidence((60, 64, 67)) is None


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_roundtrip_sources_use_manifest_then_jsonl_then_file_hash(tmp_path: Path) -> None:
    smf_dir = tmp_path / "roundtrip-smf"
    smf_dir.mkdir()
    smf = smf_dir / "kept.mid"
    smf.write_bytes(b"roundtrip")
    regenerated_hash = hashlib.sha256(smf.read_bytes()).hexdigest()
    record = {
        "name": smf.name,
        "source_sha256": hashlib.sha256(b"different-source").hexdigest(),
        "regenerated_sha256": regenerated_hash,
    }
    roundtrip = tmp_path / "roundtrip.jsonl"
    roundtrip.write_bytes(_canonical_bytes(record) + b"\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "pass",
                "outputs": {
                    "roundtrip.jsonl": hashlib.sha256(roundtrip.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )

    sources = verify_roundtrip_sources(
        root=tmp_path,
        excluded_names=frozenset(),
    )

    assert [(source.name, source.sha256) for source in sources] == [
        ("kept.mid", regenerated_hash)
    ]

    record["regenerated_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    roundtrip.write_bytes(_canonical_bytes(record) + b"\n")
    manifest.write_text(
        json.dumps(
            {
                "status": "pass",
                "outputs": {
                    "roundtrip.jsonl": hashlib.sha256(roundtrip.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="roundtrip SMF SHA-256 mismatch"):
        verify_roundtrip_sources(root=tmp_path, excluded_names=frozenset())


def _write_chord_smf(path: Path) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    for pitch in (48, 52, 55):
        track.append(mido.Message("note_on", note=pitch, velocity=64, time=0))
    for index, pitch in enumerate((48, 52, 55)):
        track.append(
            mido.Message(
                "note_off",
                note=pitch,
                velocity=0,
                time=480 if index == 0 else 0,
            )
        )
    midi.save(path)


def test_observation_run_is_compact_and_byte_deterministic(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mid"
    _write_chord_smf(source_path)
    source = ObservationDiagnosticSource(
        case_id="case",
        group_id="group",
        path=source_path,
        expected_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = run_observation_diagnostics(
        sources=(source,),
        output_dir=first,
        split_role="test",
    )
    run_observation_diagnostics(
        sources=(source,),
        output_dir=second,
        split_role="test",
    )

    assert first_result["status"] == "pass"
    assert first_result["source_count"] == 1
    assert first_result["attack_group_count"] == 1
    for name in (
        "observation-sources.jsonl",
        "harmony-support.jsonl",
        "observation-result.json",
        "observation-manifest.json",
        "run-spec.json",
        "manifest.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    record = json.loads((first / "harmony-support.jsonl").read_text(encoding="utf-8"))
    assert record["groups"][0]["attack_pitch_class_counts"] == [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    assert "harmony_support" not in record["groups"][0]


def _write_known_harmony_source(root: Path) -> KnownTimingSource:
    plan = PiecePlan(
        plan_id="plan",
        title="known harmony",
        tonal_center=0,
        mode="major",
        root_node_id="root",
        ending_intent="tonic",
        nodes=(
            PlanNode("root", None, 0, "whole"),
            PlanNode("leaf", "root", 0, "statement", duration_weight=1, score_material_id="m"),
        ),
    )
    score = ScoreSpec(
        score_id="score",
        divisions=4,
        materials=(
            ScoreMaterial(
                material_id="m",
                length_units=4,
                foreground_voice="upper",
                harmonies=(ScoreHarmony("h", 0, 4, 0, "major"),),
                notes=(
                    ScoreNote("c1", 0, 2, 48, "lower"),
                    ScoreNote("e1", 0, 2, 52, "lower"),
                    ScoreNote("g1", 0, 2, 67, "upper"),
                    ScoreNote("c2", 2, 2, 48, "lower"),
                    ScoreNote("e2", 2, 2, 64, "upper"),
                ),
            ),
        ),
    )
    performance = PerformanceSpec(
        performance_id="performance",
        target_duration_ms=4_000,
        default_velocity=64,
        timing_budget_id="subtle-v1",
        node_performances=(
            NodePerformance(
                "leaf",
                timing_profile="neutral",
                timing_amount="subtle",
                dynamics_profile="steady",
                articulation_profile="score",
                coordination_profile="aligned",
                pedal_profile="none",
            ),
        ),
    )
    root.mkdir()
    piece_path = root / "piece.dsl"
    score_path = root / "score.dsl"
    performance_path = root / "performance.dsl"
    smf_path = root / "final.mid"
    piece_path.write_text(dump_piece_plan(plan), encoding="utf-8")
    score_path.write_text(dump_score_spec(score), encoding="utf-8")
    performance_path.write_text(dump_performance_spec(performance), encoding="utf-8")
    render_performance_smf(render_performance(plan, score, performance), smf_path)
    return KnownTimingSource(
        case_id="known",
        group_id="known-family",
        root=root,
        piece_path=piece_path,
        score_path=score_path,
        performance_path=performance_path,
        smf_path=smf_path,
        evidence_kind="calibration_result",
        evidence_path=root / "unused-result.json",
        oracle_basis="generator_voice_labels",
        oracle_observation_dependency="absent",
    )


def test_oracle_annotation_runs_only_after_observation_freeze(tmp_path: Path) -> None:
    known = _write_known_harmony_source(tmp_path / "known")
    output = tmp_path / "output"
    run_observation_diagnostics(
        sources=(
            ObservationDiagnosticSource(
                case_id=known.case_id,
                group_id=known.group_id,
                path=known.smf_path,
                expected_sha256=hashlib.sha256(known.smf_path.read_bytes()).hexdigest(),
            ),
        ),
        output_dir=output,
        split_role="known_development",
    )
    frozen_hash = hashlib.sha256((output / "observation-manifest.json").read_bytes()).hexdigest()

    result = run_known_harmony_oracle(
        sources=(known,),
        observation_dir=output,
        expected_observation_manifest_sha256=frozen_hash,
    )

    assert result["status"] == "pass"
    assert result["assessed_source_count"] == 1
    annotation = json.loads((output / "oracle-annotations.jsonl").read_text(encoding="utf-8"))
    assert annotation["status"] == "assessed"
    assert annotation["declared_harmony_group_count"] == 2
    assert annotation["accompaniment_chord_tone_rate"] == {
        "numerator": 1,
        "denominator": 1,
    }

    with pytest.raises(ValueError, match="observation manifest SHA-256 mismatch"):
        run_known_harmony_oracle(
            sources=(known,),
            observation_dir=output,
            expected_observation_manifest_sha256="0" * 64,
        )
