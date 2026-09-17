from __future__ import annotations

import json
from pathlib import Path

import pytest

import llm_musical_composer.reference_structure as reference_structure
from llm_musical_composer.pilot_features import NoteEvent
from llm_musical_composer.reference_structure import (
    analyze_note_structure,
    analyze_reference_files,
    run_reference_structure_controls,
    write_reference_structure_analysis,
)


def motif(
    start: int,
    pitches: list[int],
    *,
    spacing: int = 100,
    duration: int = 180,
    velocity: int = 64,
) -> list[NoteEvent]:
    return [
        NoteEvent(pitch, start + index * spacing, duration, velocity)
        for index, pitch in enumerate(pitches)
    ]


def structured_notes() -> list[NoteEvent]:
    return (
        motif(0, [60, 64, 62, 67, 65, 69, 67, 72], velocity=45)
        + motif(1_000, [48, 55, 60, 64, 67, 72, 76, 79], velocity=100)
        + motif(2_000, [67, 71, 69, 74, 72, 76, 74, 79], velocity=45)
        + motif(3_000, [55, 59, 57, 62, 60, 64, 62, 67], velocity=70)
    )


def test_analysis_keeps_coordinates_and_resolutions_separate() -> None:
    result = analyze_note_structure(structured_notes(), resolutions=(4, 8))

    assert result["status"] == "pass"
    assert set(result["coordinates"]) == {"elapsed_time", "onset_order"}
    for coordinate in result["coordinates"].values():
        assert set(coordinate["resolutions"]) == {"4", "8"}
        assert coordinate["resolutions"]["4"]["status"] == "pass"
        assert len(coordinate["resolutions"]["4"]["windows"]) == 4
    assert "total_loss" not in result
    assert result["repetition_use"] == "diagnostic_only"


def test_sparse_high_resolution_is_unable_instead_of_zero_music() -> None:
    notes = motif(0, [60, 64, 67, 72], spacing=1_000)

    result = analyze_note_structure(notes, resolutions=(4, 32))

    for coordinate in result["coordinates"].values():
        unavailable = coordinate["resolutions"]["32"]
        assert unavailable["status"] == "unable_to_investigate"
        assert unavailable["detail"]


def test_empty_music_and_invalid_resolutions_are_explicit() -> None:
    empty = analyze_note_structure([], resolutions=(4,))

    assert empty["status"] == "unable_to_investigate"
    with pytest.raises(ValueError, match="resolutions"):
        analyze_note_structure(structured_notes(), resolutions=())
    with pytest.raises(ValueError, match="resolutions"):
        analyze_note_structure(structured_notes(), resolutions=(1,))


def test_elapsed_empty_windows_do_not_become_normal_zero_values() -> None:
    notes = motif(0, [60, 64], spacing=100) + motif(3_900, [67, 72], spacing=100)

    result = analyze_note_structure(notes, resolutions=(4,))

    elapsed = result["coordinates"]["elapsed_time"]["resolutions"]["4"]
    onset_order = result["coordinates"]["onset_order"]["resolutions"]["4"]
    assert elapsed["status"] == "unable_to_investigate"
    assert elapsed["empty_window_indices"] == [1, 2]
    assert onset_order["status"] == "pass"


def test_global_transposition_preserves_boundaries_and_peak_positions() -> None:
    notes = structured_notes()
    transposed = [
        NoteEvent(note.pitch + 5, note.onset_ms, note.duration_ms, note.velocity) for note in notes
    ]

    original = analyze_note_structure(notes, resolutions=(4, 8))
    changed = analyze_note_structure(transposed, resolutions=(4, 8))

    for coordinate_name in original["coordinates"]:
        for resolution in ("4", "8"):
            left = original["coordinates"][coordinate_name]["resolutions"][resolution]
            right = changed["coordinates"][coordinate_name]["resolutions"][resolution]
            assert (
                left["novelty"]["consensus_candidates"] == right["novelty"]["consensus_candidates"]
            )
            for name in left["activity"]:
                assert (
                    left["activity"][name]["peak_positions"]
                    == right["activity"][name]["peak_positions"]
                )


def test_reference_files_are_sorted_and_failures_remain_explicit(tmp_path: Path) -> None:
    paths = [tmp_path / "z.mid", tmp_path / "a.mid", tmp_path / "broken.mid"]
    for path in paths:
        path.write_bytes(path.name.encode())

    def loader(path: Path) -> list[NoteEvent]:
        if path.name == "broken.mid":
            raise ValueError("deliberate parse failure")
        return structured_notes()

    records = analyze_reference_files(reversed(paths), note_loader=loader, resolutions=(4,))

    assert [record["name"] for record in records] == ["a.mid", "broken.mid", "z.mid"]
    failed = records[1]
    assert failed["status"] == "unable_to_investigate"
    assert failed["error"]["type"] == "ValueError"
    assert "deliberate parse failure" in failed["error"]["message"]


def test_controls_cover_the_approved_counterexamples() -> None:
    controls = run_reference_structure_controls()

    expected = {
        "time_stretch",
        "transposition",
        "onset_jitter",
        "half_window_phase",
        "repetition_break",
        "section_reorder",
        "no_forced_four_part_form",
        "sparse_high_resolution",
    }
    assert set(controls) == expected
    assert {control["status"] for control in controls.values()} == {"pass"}


def test_written_analysis_is_deterministic_and_excludes_named_noise(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("b.mid", "rut.mid", "aimusic01.mid", "a.mid"):
        (input_dir / name).write_bytes(name.encode())

    def loader(_: Path) -> list[NoteEvent]:
        return structured_notes()

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = write_reference_structure_analysis(
        input_dir,
        first,
        note_loader=loader,
        resolutions=(4, 8),
    )
    second_summary = write_reference_structure_analysis(
        input_dir,
        second,
        note_loader=loader,
        resolutions=(4, 8),
    )

    assert first_summary == second_summary
    assert first_summary["source_count"] == 2
    assert first_summary["excluded_files"] == ["aimusic01.mid", "rut.mid"]
    first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["artifact_sha256"] == second_manifest["artifact_sha256"]
    assert set(first_manifest["artifact_sha256"]) == {
        "controls.json",
        "files.jsonl",
        "summary.json",
    }
    assert set(first_manifest["implementation_sha256"]) == {
        "reference_structure.py",
        "smf_notes.py",
        "structure_features.py",
    }


def test_empty_directory_keeps_zero_evidence_distinct_from_errors(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    summary = write_reference_structure_analysis(input_dir, tmp_path / "output")

    assert summary["source_count"] == 0
    assert summary["persistent_boundary_count"] == {
        "available_count": 0,
        "status": "unable_to_investigate",
    }


def test_failed_critical_control_stops_before_corpus_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        reference_structure,
        "run_reference_structure_controls",
        lambda: {
            "time_stretch": {"status": "fail"},
            "repetition_break": {"status": "fail"},
        },
    )

    with pytest.raises(RuntimeError, match="time_stretch"):
        write_reference_structure_analysis(tmp_path, tmp_path / "output")

    controls = json.loads((tmp_path / "output" / "controls.json").read_text(encoding="utf-8"))
    assert controls["time_stretch"]["status"] == "fail"
