from __future__ import annotations

import json
from pathlib import Path

import mido
import pytest

from llm_musical_composer.observed_structure_candidates import (
    CandidateGroup,
    ObservedStructureCandidateError,
    build_boundary_candidates,
    build_recurrence_candidates,
    canonical_recurrence_relations,
    canonical_semantic_profile_hash,
    decode_source_candidates,
    run_observed_structure_candidates,
)
from llm_musical_composer.run_state import sha256_file
from llm_musical_composer.score_timing_development_run import run_score_timing_development


def _groups(
    pitches: tuple[int, ...],
    *,
    gaps: tuple[int, ...] | None = None,
    split: int | None = None,
) -> tuple[CandidateGroup, ...]:
    if gaps is None:
        gaps = (100_000,) * (len(pitches) - 1)
    onsets = [0]
    for gap in gaps:
        onsets.append(onsets[-1] + gap)
    return tuple(
        CandidateGroup(
            group_id=f"g{index}",
            onset_us=onset,
            source_event_ids=(f"e{index}",),
            relative_onsets_us=(0,),
            pitches=((pitch + 12,) if split is not None and index >= split else (pitch,)),
            held_durations_us=(80_000,),
        )
        for index, (pitch, onset) in enumerate(zip(pitches, onsets, strict=True))
    )


def test_pause_and_texture_change_form_one_boundary_candidate() -> None:
    gaps = (100_000,) * 5 + (500_000,) + (100_000,) * 5
    groups = _groups(tuple(range(12)), gaps=gaps, split=6)

    result = build_boundary_candidates(groups, recurrences=(), window_sizes=(4,))

    assert result["status"] == "assessed"
    positions = [
        (item["start_group_index"], item["end_group_index"])
        for item in result["candidates"]
    ]
    assert positions == [(6, 6)]
    assert result["candidates"][0]["evidence_families"] == ["pause", "texture"]


def test_recurrence_candidates_are_transposition_invariant() -> None:
    pitches = (60, 62, 65, 71, 73, 67, 69, 72)
    groups = _groups(pitches)
    transposed = tuple(
        CandidateGroup(
            group_id=group.group_id,
            onset_us=group.onset_us,
            source_event_ids=group.source_event_ids,
            relative_onsets_us=group.relative_onsets_us,
            pitches=tuple(pitch + 5 for pitch in group.pitches),
            held_durations_us=group.held_durations_us,
        )
        for group in groups
    )

    first = build_recurrence_candidates(groups, tuple(range(len(groups))), window_sizes=(3,))
    second = build_recurrence_candidates(
        transposed,
        tuple(range(len(transposed))),
        window_sizes=(3,),
    )

    assert first["status"] == "assessed"
    assert first["candidates"] == second["candidates"]


def test_long_recurrence_marks_both_truncated_sides() -> None:
    motif = (0, 1, 3, 6, 10, 15, 21, 28, 36)
    groups = _groups(tuple((*motif, *(pitch + 50 for pitch in motif))))

    result = build_recurrence_candidates(
        groups,
        tuple(range(len(groups))),
        window_sizes=tuple(range(3, 9)),
    )

    length_eight = [
        item
        for item in result["candidates"]
        if "highest" in item["views"] and item["window_size"] == 8
    ]
    assert length_eight
    assert any(any(occurrence[2] for occurrence in item["occurrences"]) for item in length_eight)
    assert any(any(occurrence[1] for occurrence in item["occurrences"]) for item in length_eight)
    assert all(len(occurrence) == 3 for item in length_eight for occurrence in item["occurrences"])
    assert all("group_id" not in json.dumps(item) for item in length_eight)
    boundary = build_boundary_candidates(groups, recurrences=result["candidates"])
    recurrence_edges = [
        item["start_group_index"] for item in boundary["families"]["recurrence"]
    ]
    assert 9 in recurrence_edges
    assert 8 not in recurrence_edges
    assert 17 not in recurrence_edges


def test_identical_view_occurrences_are_compacted_without_losing_view_lineage() -> None:
    groups = _groups((60, 62, 65, 71, 73, 67, 69, 72))

    result = build_recurrence_candidates(groups, tuple(range(len(groups))), window_sizes=(3,))

    candidate = next(item for item in result["candidates"] if item["window_size"] == 3)
    assert candidate["views"] == ["highest", "lowest", "pitch_set"]
    assert set(candidate["view_pitch_shape_sha256"]) == {"highest", "lowest", "pitch_set"}
    assert candidate["occurrences"] == [[0, False, False], [5, False, False]]


def test_semantic_profile_hash_ignores_surface_profile_name() -> None:
    groups = _groups((60, 62, 64, 65))

    assert canonical_semantic_profile_hash(groups) == canonical_semantic_profile_hash(groups)


def test_dense_recurrence_does_not_qualify_a_boundary_by_itself() -> None:
    groups = _groups((60, 62, 64) * 8)
    recurrence = build_recurrence_candidates(groups, tuple(range(len(groups))))

    boundary = build_boundary_candidates(
        groups,
        recurrences=recurrence["candidates"],
        window_sizes=(4,),
    )

    assert boundary["families"]["recurrence"]
    assert boundary["status"] == "insufficient_evidence"
    assert boundary["candidates"] == []


def test_v1_and_v2_recurrence_payloads_have_the_same_canonical_relations() -> None:
    v1 = {
        "semantic_profiles": [
            {
                "semantic_profile_hash": "profile",
                "recurrence": {
                    "candidates": [
                        {
                            "view": "highest",
                            "pitch_shape_sha256": "pitch",
                            "window_size": 3,
                            "occurrences": [
                                {
                                    "start_group_index": 0,
                                    "left_extension_beyond_cap": False,
                                    "right_extension_beyond_cap": True,
                                },
                                {
                                    "start_group_index": 5,
                                    "left_extension_beyond_cap": True,
                                    "right_extension_beyond_cap": False,
                                },
                            ],
                        }
                    ]
                },
            }
        ]
    }
    v2 = {
        "schema_version": 2,
        "semantic_profiles": [
            {
                "semantic_profile_hash": "profile",
                "recurrence": {
                    "candidates": [
                        {
                            "views": ["highest"],
                            "view_pitch_shape_sha256": {"highest": "pitch"},
                            "window_size": 3,
                            "occurrences": [[0, False, True], [5, True, False]],
                        }
                    ]
                },
            }
        ],
    }

    assert canonical_recurrence_relations(v1) == canonical_recurrence_relations(v2)


def _write_midi(path: Path, *, shift: int) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    pitches = (60, 62, 65, 71, 73, 67, 69, 72, 75, 81, 83, 86)
    for index, pitch in enumerate(pitches):
        track.append(
            mido.Message(
                "note_on",
                note=pitch + shift,
                velocity=64,
                channel=0,
                time=480 if index == 6 else 120,
            )
        )
        track.append(
            mido.Message("note_off", note=pitch + shift, velocity=0, channel=0, time=60)
        )
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(path)


def _development_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    staging = tmp_path / "development-smf"
    staging.mkdir()
    _write_midi(staging / "alpha.mid", shift=0)
    _write_midi(staging / "beta.mid", shift=2)
    staging_manifest = tmp_path / "staging-manifest.json"
    staging_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "development_smf": {
                    path.name: sha256_file(path) for path in sorted(staging.iterdir())
                },
            }
        ),
        encoding="utf-8",
    )
    timing = tmp_path / "score-timing"
    run_score_timing_development(
        staging_dir=staging,
        staging_manifest=staging_manifest,
        output_dir=timing,
    )
    return staging, staging_manifest, timing / "files.jsonl", timing / "manifest.json"


def test_development_runner_writes_compact_deterministic_artifacts(tmp_path: Path) -> None:
    staging, staging_manifest, timing_files, timing_manifest = _development_artifacts(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_result = run_observed_structure_candidates(
        staging_dir=staging,
        staging_manifest=staging_manifest,
        score_timing_files=timing_files,
        score_timing_manifest=timing_manifest,
        output_dir=first,
    )
    second_result = run_observed_structure_candidates(
        staging_dir=staging,
        staging_manifest=staging_manifest,
        score_timing_files=timing_files,
        score_timing_manifest=timing_manifest,
        output_dir=second,
    )

    assert first_result["status"] == "pass"
    assert first_result == second_result
    assert first_result["source_count"] == 2
    for name in (
        "source-candidates.jsonl",
        "negative-controls.jsonl",
        "summary.json",
        "run-spec.json",
        "manifest.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    payload = (first / "source-candidates.jsonl").read_text(encoding="utf-8")
    assert '"notes"' not in payload
    assert '"events"' not in payload
    assert '"piece_plan"' not in payload
    records = [json.loads(line) for line in payload.splitlines()]
    assert all(record["schema_version"] == 2 for record in records)
    for record in records:
        for profile in record["semantic_profiles"]:
            for candidate in profile["boundary"]["candidates"]:
                assert "anchor" in candidate
                assert "start_anchor" not in candidate
                assert "end_anchor" not in candidate
    lower = {
        record["name"]: record
        for record in (
            json.loads(line) for line in timing_files.read_text(encoding="utf-8").splitlines()
        )
    }
    for record in records:
        decoded = decode_source_candidates(record, lower[record["name"]])
        member_count = sum(len(alias["members"]) for alias in decoded["aliases"])
        assert member_count == record["score_timing_candidate_count"]
        assert set(decoded["eligible_recurrence_edges"]) == {
            profile["semantic_profile_hash"] for profile in record["semantic_profiles"]
        }


def test_development_runner_rejects_modified_score_timing_input(tmp_path: Path) -> None:
    staging, staging_manifest, timing_files, timing_manifest = _development_artifacts(tmp_path)
    timing_files.write_text(timing_files.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ObservedStructureCandidateError, match="SHA-256 mismatch"):
        run_observed_structure_candidates(
            staging_dir=staging,
            staging_manifest=staging_manifest,
            score_timing_files=timing_files,
            score_timing_manifest=timing_manifest,
            output_dir=tmp_path / "output",
        )


def test_runner_preserves_generic_staging_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "holdout-smf"
    staging.mkdir()
    _write_midi(staging / "alpha.mid", shift=0)
    staging_manifest = tmp_path / "staging-manifest.json"
    staging_manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "pass",
                "split_role": "holdout",
                "inputs": {"method_manifest": "f" * 64},
                "staged_smf": {
                    path.name: sha256_file(path) for path in sorted(staging.iterdir())
                },
            }
        ),
        encoding="utf-8",
    )
    method_manifest = tmp_path / "method-manifest.json"
    method_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "llm_musical_composer.score_timing_development_run.verify_method_manifest",
        lambda **_arguments: "f" * 64,
    )
    monkeypatch.setattr(
        "llm_musical_composer.observed_structure_candidates.verify_method_manifest",
        lambda **_arguments: "f" * 64,
    )
    timing = tmp_path / "score-timing"
    run_score_timing_development(
        staging_dir=staging,
        staging_manifest=staging_manifest,
        output_dir=timing,
        repository_root=Path.cwd(),
        method_manifest=method_manifest,
    )

    output = tmp_path / "output"
    run_observed_structure_candidates(
        staging_dir=staging,
        staging_manifest=staging_manifest,
        score_timing_files=timing / "files.jsonl",
        score_timing_manifest=timing / "manifest.json",
        output_dir=output,
        repository_root=Path.cwd(),
        method_manifest=method_manifest,
    )

    run_spec = json.loads((output / "run-spec.json").read_text(encoding="utf-8"))
    assert run_spec["split_role"] == "holdout"
    assert "staged_smf" in run_spec["inputs"]
    assert "development_smf" not in run_spec["inputs"]
    assert run_spec["inputs"]["method_manifest"] == "f" * 64
