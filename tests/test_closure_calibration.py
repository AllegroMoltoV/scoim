from __future__ import annotations

import json
from pathlib import Path

import mido
import pytest

from llm_musical_composer.closure_calibration import (
    ALLOWED_ANSWERS,
    Candidate,
    PedalEvent,
    Performance,
    assign_pair_roles,
    assign_splits,
    build_calibration_set,
    build_conflicts,
    create_evaluation_form,
    find_natural_middle_end,
    read_performance,
    select_diverse_candidates,
    slice_performance,
    summarize_performance,
    validate_answer,
    write_performance,
)
from llm_musical_composer.pilot_features import NoteEvent


def _save_performance(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=500)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=4_000))
    piano = mido.MidiTrack()
    events = [
        (0, mido.Message("control_change", channel=0, control=64, value=90)),
        (0, mido.Message("note_on", channel=0, note=60, velocity=70)),
        (1_000, mido.Message("note_off", channel=0, note=60, velocity=0)),
        (1_500, mido.Message("note_on", channel=0, note=64, velocity=80)),
        (2_500, mido.Message("note_off", channel=0, note=64, velocity=0)),
        (3_000, mido.Message("note_on", channel=9, note=36, velocity=100)),
        (3_100, mido.Message("note_off", channel=9, note=36, velocity=0)),
        (3_500, mido.Message("control_change", channel=0, control=64, value=0)),
    ]
    previous = 0
    for tick, message in events:
        message.time = tick - previous
        piano.append(message)
        previous = tick
    piano.append(mido.MetaMessage("end_of_track", time=500))
    midi.tracks.extend((meta, piano))
    midi.save(path)


def test_read_slice_and_write_performance_closes_boundary_state(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    _save_performance(source)

    performance = read_performance(source)
    excerpt = slice_performance(performance, 500, 2_000)
    output = tmp_path / "excerpt.mid"
    write_performance(excerpt, output)
    reread = read_performance(output)

    assert [(note.pitch, note.onset_ms, note.duration_ms) for note in excerpt.notes] == [
        (60, 0, 500),
        (64, 1_000, 500),
    ]
    assert excerpt.pedals[0] == PedalEvent(0, 90)
    assert excerpt.pedals[-1] == PedalEvent(1_500, 0)
    assert reread.duration_ms == 1_500
    assert reread.notes == excerpt.notes
    assert reread.pedals[-1].value == 0
    assert all(note.onset_ms + note.duration_ms <= excerpt.duration_ms for note in excerpt.notes)


def test_read_performance_rejects_format_two_and_empty_slice(tmp_path: Path) -> None:
    midi = mido.MidiFile(type=2, ticks_per_beat=500)
    midi.tracks.append(mido.MidiTrack())
    path = tmp_path / "format-two.mid"
    midi.save(path)

    with pytest.raises(ValueError, match="format 2"):
        read_performance(path)
    with pytest.raises(ValueError, match="end must be greater"):
        slice_performance(Performance((), (), 100), 50, 50)


def test_natural_middle_end_stays_in_search_range_and_prefers_low_activity() -> None:
    notes = (
        NoteEvent(60, 0, 1_000, 70),
        NoteEvent(62, 2_000, 1_000, 70),
        NoteEvent(64, 5_800, 200, 70),
        NoteEvent(67, 6_050, 400, 70),
        NoteEvent(69, 6_900, 100, 70),
        NoteEvent(71, 9_000, 1_000, 70),
    )
    performance = Performance(notes, (), 10_000)

    ending = find_natural_middle_end(performance)

    assert ending == 7_000
    assert 6_000 <= ending <= 7_000
    assert find_natural_middle_end(Performance((), (), 10_000)) is None


def test_conflicts_and_diverse_selection_are_deterministic() -> None:
    evaluation_groups = {"filename_series": [{"series": "set", "files": ["a.mid", "b.mid"]}]}
    duplicate_candidates = {
        "exact_groups": [{"files": ["c.mid", "d.mid"]}],
        "musical_fingerprint_groups": [],
        "pairwise_positive_evidence": [{"first": "e.mid", "second": "f.mid"}],
    }
    conflicts = build_conflicts(evaluation_groups, duplicate_candidates)
    candidates = [
        Candidate(f"{name}.mid", Path(f"{name}.mid"), (float(index), float(index % 2)))
        for index, name in enumerate("abcdef", start=1)
    ]

    first = select_diverse_candidates(candidates, conflicts, count=3, seed=41)
    second = select_diverse_candidates(candidates, conflicts, count=3, seed=41)

    assert [item.name for item in first] == [item.name for item in second]
    selected = {item.name for item in first}
    assert not {"a.mid", "b.mid"} <= selected
    assert not {"c.mid", "d.mid"} <= selected
    assert not {"e.mid", "f.mid"} <= selected
    with pytest.raises(ValueError, match="cannot select"):
        select_diverse_candidates(candidates[:2], conflicts, count=2, seed=41)


def test_pair_roles_and_splits_are_stable_and_balanced() -> None:
    pair_ids = [f"pair-{index:02d}" for index in range(16)]

    assert assign_pair_roles("song.mid", seed=8) == assign_pair_roles("song.mid", seed=8)
    splits = assign_splits(pair_ids)

    assert set(splits) == set(pair_ids)
    assert list(splits.values()).count("few_shot") == 8
    assert list(splits.values()).count("holdout") == 8
    with pytest.raises(ValueError, match="even"):
        assign_splits(pair_ids[:-1])


def test_summary_and_evaluation_form_use_only_declared_values(tmp_path: Path) -> None:
    performance = Performance(
        (
            NoteEvent(60, 0, 500, 40),
            NoteEvent(67, 750, 250, 90),
        ),
        (PedalEvent(0, 80), PedalEvent(1_000, 0)),
        1_000,
    )

    summary = summarize_performance(performance, window_count=4, terminal_limit=1)
    form_path = tmp_path / "evaluation-form.json"
    create_evaluation_form(["pair-01", "pair-02"], form_path)
    form = json.loads(form_path.read_text(encoding="utf-8"))

    assert sum(window["onset_count"] for window in summary["windows"]) == 2
    assert len(summary["terminal_events"]) == 1
    assert summary["terminal_events"][0] == {
        "onset_ratio": 0.75,
        "duration_ratio": 0.25,
        "onset_ms": 750,
        "duration_ms": 250,
        "pitch": 67,
        "velocity": 90,
    }
    assert summary["pedal_events"][-1]["at_ratio"] == 1.0
    assert summary["pedal_events"][-1]["at_ms"] == 1_000
    assert summary["pedal_events"][-1]["value"] == 0
    assert form["allowed_values"] == list(ALLOWED_ANSWERS)
    assert form["scope"] == "局所的な終止感"
    assert [item["pair_id"] for item in form["answers"]] == ["pair-01", "pair-02"]
    assert all(item["answer"] is None for item in form["answers"])
    for value in ALLOWED_ANSWERS:
        assert validate_answer(value) == value
    with pytest.raises(ValueError, match="allowed"):
        validate_answer("Aが好き")


def test_build_calibration_set_writes_anonymous_deterministic_artifacts(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    audit_root = tmp_path / "audit"
    output_root = tmp_path / "output"
    source_root.mkdir()
    audit_root.mkdir()
    records = []
    for file_index, name in enumerate(("one.mid", "two.mid", "three.mid", "four.mid")):
        performance = Performance(
            tuple(
                NoteEvent(
                    48 + file_index * 3 + note_index % 12,
                    note_index * 500,
                    300 + file_index * 10,
                    50 + note_index % 40,
                )
                for note_index in range(21)
            ),
            (PedalEvent(0, 90), PedalEvent(10_500, 0)),
            10_500,
        )
        write_performance(performance, source_root / name)
        records.append(
            {
                "name": name,
                "relative_path": name,
                "status": "affirmative_evidence",
                "conformance": {"is_structurally_conformant": True},
            }
        )
    records.append(
        {
            "name": "rut.mid",
            "relative_path": "rut.mid",
            "status": "affirmative_evidence",
            "conformance": {"is_structurally_conformant": True},
        }
    )
    (audit_root / "files.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (audit_root / "evaluation-groups.json").write_text(
        json.dumps({"filename_series": []}), encoding="utf-8"
    )
    (audit_root / "duplicate-candidates.json").write_text(
        json.dumps(
            {
                "exact_groups": [],
                "musical_fingerprint_groups": [],
                "pairwise_positive_evidence": [],
            }
        ),
        encoding="utf-8",
    )

    result = build_calibration_set(source_root, audit_root, output_root, count=2, seed=77)
    manifest = json.loads((output_root / "selection-manifest.json").read_text(encoding="utf-8"))
    judge = json.loads((output_root / "judge" / "pair-01.json").read_text(encoding="utf-8"))

    assert result["pair_count"] == 2
    assert result["diagnostics"]["excluded_by_user"] == ["rut.mid"]
    assert len(manifest["pairs"]) == 2
    assert {item["split"] for item in manifest["pairs"]} == {"few_shot", "holdout"}
    assert set(manifest["pairs"][0]["roles"]) == {"A", "B"}
    assert (output_root / "listen" / "pair-01" / "sample-A.mid").is_file()
    assert (output_root / "listen" / "pair-01" / "sample-B.mid").is_file()
    assert (output_root / "judge" / "pair-01.png").read_bytes().startswith(b"\x89PNG")
    assert "one.mid" not in json.dumps(judge)
    assert "two.mid" not in json.dumps(judge)
    assert set(judge["samples"]) == {"A", "B"}
    assert judge["scope"] == "局所的な終止感"
    for label in ("A", "B"):
        reread = read_performance(output_root / "listen" / "pair-01" / f"sample-{label}.mid")
        assert reread.notes
        assert reread.pedals[-1].value == 0
