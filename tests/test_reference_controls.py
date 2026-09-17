from pathlib import Path

import pytest

from llm_musical_composer.reference_controls import (
    EXCLUDED_NAMES,
    build_reference_profile_v1,
    destroy_timing,
    evaluate_reference_controls,
    main,
    stretch_piece,
    transpose_piece,
)
from llm_musical_composer.reference_profile import ReferencePiece, ReferenceProfileError


def _piece(name: str, offset: int, rhythm_scale: int = 1) -> ReferencePiece:
    notes = []
    onsets = [
        value + index * index * abs(offset) * 7
        for index, value in enumerate([0, 300, 800, 1200, 2100, 2600, 3400, 4300])
    ]
    pitches = [48, 55, 52, 60, 57, 64, 59, 67]
    for index, (onset, pitch) in enumerate(zip(onsets, pitches, strict=True)):
        contour_change = ((index % 3) - 1) * (abs(offset) % 4)
        notes.append(
            {
                "pitch": pitch + offset + contour_change,
                "onset_ms": onset * rhythm_scale,
                "duration_ms": (220 + index * 45) * rhythm_scale,
                "velocity": 55 + index * 8 + (offset if index % 2 else 0),
            }
        )
        if index % 3 == 0:
            notes.append(
                {
                    "pitch": pitch + offset + contour_change + 7 + (abs(offset) % 3),
                    "onset_ms": onset * rhythm_scale,
                    "duration_ms": (400 + index * 20) * rhythm_scale,
                    "velocity": 58 + index * 8,
                }
            )
    return ReferencePiece.from_dicts(
        name=name,
        notes=notes,
        pedals=[
            {"at_ms": 0, "value": 127},
            {"at_ms": 1900 * rhythm_scale, "value": 0},
            {"at_ms": 2050 * rhythm_scale, "value": 127},
            {"at_ms": 5000 * rhythm_scale, "value": 0},
        ],
    )


def test_reference_controls_keep_groups_separate_and_pass_synthetic_controls() -> None:
    pieces = [
        _piece("a.mid", 0),
        _piece("b.mid", 2),
        _piece("c.mid", -3),
        _piece("d.mid", 5),
    ]

    report = evaluate_reference_controls(pieces)

    assert report["status"] == "pass"
    assert report["passing_group_count"] >= 2
    assert set(report["groups"]) == {
        "performance_texture",
        "rhythm_time",
        "pitch_harmony",
    }
    assert "overall_style_loss" not in report
    assert report["controls"]["input_order"]["status"] == "pass"


def test_control_evaluation_requires_three_unique_pieces() -> None:
    with pytest.raises(ReferenceProfileError, match="three unique"):
        evaluate_reference_controls([_piece("a.mid", 0), _piece("a.mid", 2)])


def test_transform_arguments_and_midi_range_are_validated() -> None:
    piece = _piece("a.mid", 0)
    with pytest.raises(ReferenceProfileError, match="integer"):
        transpose_piece(piece, True)
    with pytest.raises(ReferenceProfileError, match="MIDI pitch"):
        transpose_piece(piece, 100)
    for factor in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ReferenceProfileError, match="finite and positive"):
            stretch_piece(piece, factor)
    assert destroy_timing(piece).notes != piece.notes


def test_controls_can_report_absent_pedal_without_turning_it_into_distance_zero_failure() -> None:
    pieces = [
        ReferencePiece(piece.name, piece.notes, ())
        for piece in (_piece("a.mid", 0), _piece("b.mid", 2), _piece("c.mid", -3))
    ]

    report = evaluate_reference_controls(pieces)

    assert report["controls"]["pedal_removal"]["status"] == "unable_to_investigate"
    assert report["status"] == "pass"


def test_corpus_builder_excludes_known_noise_and_records_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for name in ("good-a.mid", "good-b.mid", "good-c.mid", "rut.mid", "aimusic01.mid", "bad.mid"):
        (source_dir / name).write_bytes(b"placeholder")
    pieces = {
        name: _piece(name, offset)
        for name, offset in (("good-a.mid", 0), ("good-b.mid", 2), ("good-c.mid", -2))
    }

    def fake_load(path: Path) -> ReferencePiece:
        if path.name == "bad.mid":
            raise ReferenceProfileError("broken")
        if path.name in EXCLUDED_NAMES:
            raise AssertionError("excluded files must not be loaded")
        return pieces[path.name]

    monkeypatch.setattr("llm_musical_composer.reference_controls.load_reference_piece", fake_load)
    output_dir = tmp_path / "output"

    summary = build_reference_profile_v1(
        source_dir=source_dir, output_dir=output_dir, neighbor_count=3
    )

    assert summary["source_count"] == 3
    assert {item["name"] for item in summary["excluded"]} == EXCLUDED_NAMES
    assert summary["unable_to_investigate"] == [
        {"name": "bad.mid", "status": "unable_to_investigate", "error": "broken"}
    ]
    assert summary["long_term_structure"] == {
        "status": "unverified",
        "included_in_similarity": False,
    }
    assert (output_dir / "files.jsonl").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "manifest.json").is_file()


def test_corpus_builder_rejects_too_few_usable_references(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    output_dir = tmp_path / "output"

    with pytest.raises(ReferenceProfileError, match="not enough usable"):
        build_reference_profile_v1(source_dir=source_dir, output_dir=output_dir, neighbor_count=3)


def test_reference_profile_cli_reports_pass_and_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = {
        "status": "pass",
        "source_count": 3,
        "controls": {"passing_group_count": 3},
        "default_reference": {"selected": {"name": "a.mid"}},
    }
    monkeypatch.setattr(
        "llm_musical_composer.reference_controls.build_reference_profile_v1",
        lambda **kwargs: result,
    )
    arguments = ["--source-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")]

    assert main(arguments) == 0
    assert '"status": "pass"' in capsys.readouterr().out
    result["status"] = "fail"
    assert main(arguments) == 2
