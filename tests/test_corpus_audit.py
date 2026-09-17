from __future__ import annotations

import json
import struct
from pathlib import Path

import mido
import pytest

from llm_musical_composer.corpus_audit import (
    InvalidSmfStructureError,
    audit_corpus,
    audit_file,
    build_musical_fingerprint,
    fingerprint_similarity,
    inspect_smf_bytes,
    main,
)


def save_midi(path: Path, *, ticks_per_beat: int = 480, transpose: int = 0) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    meta_track = mido.MidiTrack()
    note_track = mido.MidiTrack()
    midi.tracks.extend([meta_track, note_track])

    meta_track.append(mido.MetaMessage("track_name", name="Meta", time=0))
    meta_track.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0))
    meta_track.append(mido.MetaMessage("key_signature", key="C", time=0))
    meta_track.append(mido.MetaMessage("set_tempo", tempo=600_000, time=0))
    meta_track.append(mido.MetaMessage("end_of_track", time=ticks_per_beat * 3))

    note_track.append(mido.MetaMessage("track_name", name="Piano", time=0))
    note_track.append(mido.Message("program_change", channel=0, program=0, time=0))
    note_track.append(mido.Message("control_change", channel=0, control=64, value=127, time=0))
    for pitch, velocity in ((60, 64), (64, 80), (67, 96)):
        note_track.append(
            mido.Message(
                "note_on",
                channel=0,
                note=pitch + transpose,
                velocity=velocity,
                time=0,
            )
        )
        note_track.append(
            mido.Message(
                "note_off",
                channel=0,
                note=pitch + transpose,
                velocity=32,
                time=ticks_per_beat,
            )
        )
    note_track.append(mido.Message("control_change", channel=0, control=64, value=0, time=0))
    note_track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(path)


def test_audit_file_reports_structure_and_expression(tmp_path: Path) -> None:
    path = tmp_path / "piece.mid"
    save_midi(path)

    result = audit_file(path)

    assert result["status"] == "affirmative_evidence"
    assert result["header"] == {
        "signature": "MThd",
        "header_length": 6,
        "format": 1,
        "declared_tracks": 2,
        "timing_kind": "ppqn",
        "ticks_per_beat": 480,
    }
    assert result["structure"]["parsed_tracks"] == 2
    assert result["structure"]["note_count"] == 3
    assert result["structure"]["pitch_min"] == 60
    assert result["structure"]["pitch_max"] == 67
    assert result["structure"]["max_polyphony"] == 1
    assert result["structure"]["estimated_measures"] == pytest.approx(1.0)
    assert result["performance"]["velocity_unique"] == 3
    assert result["performance"]["sustain_event_count"] == 2
    assert result["performance"]["other_control_change_count"] == 0
    assert result["anomalies"]["dangling_note_on"] == 0
    assert result["anomalies"]["unmatched_note_off"] == 0
    assert result["maps"]["time_signatures"][0]["numerator"] == 3
    assert result["maps"]["tempos"][0]["microseconds_per_beat"] == 600_000
    assert result["fingerprint"]["token_count"] == 3


def test_audit_file_distinguishes_note_anomalies_and_empty_song(tmp_path: Path) -> None:
    path = tmp_path / "anomalies.mid"
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("note_off", note=62, velocity=0, time=0))
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_on", note=60, velocity=90, time=10))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(path)

    result = audit_file(path)

    assert result["anomalies"]["unmatched_note_off"] == 1
    assert result["anomalies"]["overlapping_note_on"] == 1
    assert result["anomalies"]["dangling_note_on"] == 2
    assert result["structure"]["note_count"] == 2
    assert result["structure"]["max_polyphony"] == 2

    empty_path = tmp_path / "empty.mid"
    empty = mido.MidiFile(type=0, ticks_per_beat=480)
    empty.tracks.append(mido.MidiTrack([mido.MetaMessage("end_of_track", time=0)]))
    empty.save(empty_path)
    empty_result = audit_file(empty_path)
    assert empty_result["anomalies"]["empty_song"] is True


def test_audit_file_preserves_failures_instead_of_returning_zero(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mid"
    broken.write_bytes(b"not-midi")

    result = audit_file(broken)

    assert result["status"] == "unable_to_investigate"
    assert result["error"]["type"]
    assert result["error"]["message"]
    assert result["structure"] is None


def test_audit_file_identifies_smpte_without_parsing_as_ppqn(tmp_path: Path) -> None:
    path = tmp_path / "smpte.mid"
    division = 0xE728
    track_data = b"\x00\xff\x2f\x00"
    path.write_bytes(
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, division)
        + b"MTrk"
        + struct.pack(">I", len(track_data))
        + track_data
    )

    result = audit_file(path)

    assert result["status"] == "unable_to_investigate"
    assert result["header"]["timing_kind"] == "smpte"
    assert result["error"]["type"] == "UnsupportedTimingError"


def test_byte_inspection_reports_noncanonical_and_trailing_structures() -> None:
    track_data = b"\x00\xff\x2f\x00"
    data = (
        b"MThd"
        + struct.pack(">IHHH", 7, 0, 2, 480)
        + b"\x00"
        + b"MTrk"
        + struct.pack(">I", len(track_data))
        + track_data
        + b"MTrk"
        + struct.pack(">I", len(track_data))
        + track_data
        + b"tail"
    )

    inspection = inspect_smf_bytes(data)

    codes = {issue["code"] for issue in inspection.issues}
    assert codes == {
        "format_track_count_mismatch",
        "noncanonical_header_length",
        "trailing_data",
    }
    assert inspection.trailing_bytes == 4
    assert len(inspection.track_chunks) == 2


def test_byte_inspection_reports_zero_tracks_and_invalid_smpte() -> None:
    zero_tracks = inspect_smf_bytes(b"MThd" + struct.pack(">IHHH", 6, 1, 0, 480))
    assert [issue["code"] for issue in zero_tracks.issues] == ["zero_declared_tracks"]

    invalid_smpte = inspect_smf_bytes(
        b"MThd" + struct.pack(">IHHH", 6, 0, 1, 0xFF00) + b"MTrk" + struct.pack(">I", 0)
    )
    assert [issue["code"] for issue in invalid_smpte.issues] == ["invalid_smpte_division"]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "shorter"),
        (b"xxxx" + struct.pack(">I", 6), "not MThd"),
        (b"MThd" + struct.pack(">I", 5) + b"\x00" * 5, "shorter than 6"),
        (b"MThd" + struct.pack(">I", 6) + b"\x00", "truncated"),
        (b"MThd" + struct.pack(">IHHH", 6, 3, 1, 480), "format value"),
        (
            b"MThd" + struct.pack(">IHHH", 6, 1, 1, 480),
            "chunk header is missing",
        ),
        (
            b"MThd" + struct.pack(">IHHH", 6, 1, 1, 480) + b"NOPE" + struct.pack(">I", 0),
            "does not start with MTrk",
        ),
        (
            b"MThd" + struct.pack(">IHHH", 6, 1, 1, 480) + b"MTrk" + struct.pack(">I", 10),
            "beyond end of file",
        ),
    ],
)
def test_byte_inspection_rejects_invalid_structures(data: bytes, message: str) -> None:
    with pytest.raises(InvalidSmfStructureError, match=message):
        inspect_smf_bytes(data)


def test_audit_file_marks_format_two_global_metrics_not_applicable(tmp_path: Path) -> None:
    path = tmp_path / "format2.mid"
    midi = mido.MidiFile(type=2, ticks_per_beat=480)
    for pitch in (60, 67):
        track = mido.MidiTrack()
        track.append(mido.Message("note_on", note=pitch, velocity=80, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=480))
        midi.tracks.append(track)
    midi.save(path)

    result = audit_file(path)

    assert result["status"] == "affirmative_evidence"
    assert result["structure"]["duration_seconds"] is None
    assert result["structure"]["estimated_measures"] is None
    assert result["structure"]["notes_per_quarter"] is None
    assert "format_2_global_metrics_not_applicable" in {
        issue["code"] for issue in result["conformance"]["issues"]
    }


def test_audit_file_reports_missing_end_of_track_and_parser_failure(tmp_path: Path) -> None:
    missing_eot = tmp_path / "missing-eot.mid"
    track_data = b"\x00\x90\x3c\x40"
    missing_eot.write_bytes(
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, 480)
        + b"MTrk"
        + struct.pack(">I", len(track_data))
        + track_data
    )
    result = audit_file(missing_eot)
    assert result["status"] == "affirmative_evidence"
    assert "missing_end_of_track" in {issue["code"] for issue in result["conformance"]["issues"]}

    invalid_event = tmp_path / "invalid-event.mid"
    invalid_track = b"\x00\x90\xff\x40"
    invalid_event.write_bytes(
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, 480)
        + b"MTrk"
        + struct.pack(">I", len(invalid_track))
        + invalid_track
    )
    failed = audit_file(invalid_event)
    assert failed["status"] == "unable_to_investigate"
    assert failed["header"]["format"] == 0


def test_audit_file_reports_multiple_eot_and_message_categories(tmp_path: Path) -> None:
    multiple_eot = tmp_path / "multiple-eot.mid"
    track_data = b"\x00\xff\x2f\x00\x00\x90\x3c\x40\x00\xff\x2f\x00"
    multiple_eot.write_bytes(
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, 480)
        + b"MTrk"
        + struct.pack(">I", len(track_data))
        + track_data
    )
    result = audit_file(multiple_eot)
    codes = {issue["code"] for issue in result["conformance"]["issues"]}
    assert {"data_after_end_of_track", "multiple_end_of_track"} <= codes

    categories = tmp_path / "categories.mid"
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("control_change", control=1, value=32, time=0))
    track.append(mido.Message("pitchwheel", pitch=100, time=0))
    track.append(mido.Message("aftertouch", value=20, time=0))
    track.append(mido.Message("polytouch", note=60, value=30, time=0))
    track.append(mido.Message("sysex", data=(1, 2, 3), time=0))
    track.append(mido.Message("note_on", note=60, velocity=70, time=0))
    track.append(mido.Message("note_on", note=60, velocity=0, time=480))
    midi.save(categories)

    category_result = audit_file(categories)
    assert category_result["maps"]["tempo_default_assumed"] is True
    assert category_result["maps"]["time_signature_default_assumed"] is True
    assert category_result["performance"]["other_control_change_count"] == 1
    assert category_result["performance"]["pitchwheel_event_count"] == 1
    assert category_result["performance"]["aftertouch_event_count"] == 2
    assert category_result["performance"]["sysex_event_count"] == 1
    assert category_result["anomalies"]["dangling_note_on"] == 0


def test_fingerprint_is_invariant_to_transposition_tempo_and_ppqn(tmp_path: Path) -> None:
    first = tmp_path / "first.mid"
    second = tmp_path / "second.mid"
    save_midi(first, ticks_per_beat=480, transpose=0)
    save_midi(second, ticks_per_beat=960, transpose=5)

    first_result = audit_file(first)
    second_result = audit_file(second)

    assert first_result["fingerprint"]["sha256"] == second_result["fingerprint"]["sha256"]
    assert (
        fingerprint_similarity(
            first_result["fingerprint"]["ngrams"],
            second_result["fingerprint"]["ngrams"],
        )
        == 1.0
    )


def test_build_musical_fingerprint_handles_short_and_empty_sequences() -> None:
    empty = build_musical_fingerprint([], ticks_per_beat=480)
    assert empty["token_count"] == 0
    assert empty["ngrams"] == []
    assert fingerprint_similarity([], []) == 1.0
    assert fingerprint_similarity(["a"], []) == 0.0
    assert fingerprint_similarity(["a"], ["b"]) == 0.0


def test_audit_corpus_writes_structured_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    save_midi(input_dir / "series1.mid")
    save_midi(input_dir / "series2.mid")
    (input_dir / "broken.mid").write_bytes(b"broken")
    (input_dir / "asset.musicloop").write_bytes(b"not-an-smf")
    (input_dir / "unexpected").mkdir()

    summary = audit_corpus(input_dir, output_dir)

    assert summary["input_file_count"] == 5
    assert summary["midi_file_count"] == 3
    assert summary["affirmative_evidence_count"] == 2
    assert summary["unable_to_investigate_count"] == 2
    assert summary["excluded_non_midi_count"] == 1
    assert summary["exact_duplicate_group_count"] == 1
    assert summary["distributions"]["formats"] == {"1": 2}
    assert summary["distributions"]["note_count"]["median"] == 3.0
    assert summary["feature_file_counts"]["sustain_present"] == 2
    assert summary["feature_file_counts"]["explicit_tempo"] == 2
    assert summary["feature_file_counts"]["explicit_program_change"] == 2
    assert summary["channel_file_counts"] == {"0": 2}
    assert summary["anomaly_totals"]["empty_song"] == 0

    lines = (output_dir / "files.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert {record["name"] for record in records} == {
        "asset.musicloop",
        "broken.mid",
        "series1.mid",
        "series2.mid",
        "unexpected",
    }

    duplicate_candidates = json.loads(
        (output_dir / "duplicate-candidates.json").read_text(encoding="utf-8")
    )
    assert duplicate_candidates["exact_groups"][0]["files"] == [
        "series1.mid",
        "series2.mid",
    ]
    assert duplicate_candidates["musical_fingerprint_groups"][0]["files"] == [
        "series1.mid",
        "series2.mid",
    ]
    assert "near_candidate_minimum_jaccard" not in duplicate_candidates
    assert summary["pairwise_compared_count"] == 1
    assert duplicate_candidates["top_candidates_per_file"]["series1.mid"] == [
        {"file": "series2.mid", "jaccard": 1.0}
    ]

    groups = json.loads((output_dir / "evaluation-groups.json").read_text(encoding="utf-8"))
    assert groups["filename_series"][0]["series"] == "series"
    assert groups["track_configuration"]["multi_track"] == ["series1.mid", "series2.mid"]


def test_audit_corpus_requires_an_existing_input_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        audit_corpus(tmp_path / "missing", tmp_path / "output")


def test_audit_corpus_groups_japanese_numbered_movements(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    save_midi(input_dir / "組曲「水」第1曲「泉」.mid")
    save_midi(input_dir / "組曲「水」第2曲「川」.mid")

    audit_corpus(input_dir, output_dir)

    groups = json.loads((output_dir / "evaluation-groups.json").read_text(encoding="utf-8"))
    assert groups["filename_series"] == [
        {
            "series": "組曲「水」",
            "files": ["組曲「水」第1曲「泉」.mid", "組曲「水」第2曲「川」.mid"],
        }
    ]


def test_main_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    save_midi(input_dir / "piece.mid")

    assert main([str(input_dir), str(output_dir)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["affirmative_evidence_count"] == 1
