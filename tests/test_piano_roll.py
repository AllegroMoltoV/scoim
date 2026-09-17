from __future__ import annotations

import struct
from pathlib import Path

import pytest

from llm_musical_composer.closure_calibration import PedalEvent, Performance
from llm_musical_composer.piano_roll import render_pair_piano_roll
from llm_musical_composer.pilot_features import NoteEvent


def test_pair_piano_roll_writes_png_and_reports_matching_rectangles(tmp_path: Path) -> None:
    first = Performance(
        (NoteEvent(60, 0, 500, 40), NoteEvent(72, 500, 500, 100)),
        (PedalEvent(0, 100), PedalEvent(1_000, 0)),
        1_000,
    )
    second = Performance((NoteEvent(55, 250, 500, 80),), (), 1_000)
    output = tmp_path / "pair.png"

    metadata = render_pair_piano_roll(first, second, output, width=640, height=300)
    data = output.read_bytes()

    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", data[16:24]) == (640, 300)
    assert metadata["samples"]["A"]["note_rectangles"] == 2
    assert metadata["samples"]["B"]["note_rectangles"] == 1
    assert metadata["pitch_range"] == [55, 72]


def test_pair_piano_roll_rejects_invalid_canvas_and_empty_notes(tmp_path: Path) -> None:
    empty = Performance((), (), 1_000)
    note = Performance((NoteEvent(60, 0, 100, 60),), (), 100)

    with pytest.raises(ValueError, match="canvas"):
        render_pair_piano_roll(note, note, tmp_path / "small.png", width=100, height=100)
    with pytest.raises(ValueError, match="pitched note"):
        render_pair_piano_roll(empty, empty, tmp_path / "empty.png")
