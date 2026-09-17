import mido
import pytest

from scoim.neutral_score_preview import (
    NeutralPreviewSegment,
    check_neutral_score_preview,
    write_neutral_score_preview,
)
from scoim.score_ir import ScoreNote


def test_neutral_score_preview_rejects_a_changed_note_event(tmp_path) -> None:
    segments = (
        NeutralPreviewSegment(
            12,
            (
                ScoreNote("upper", 0, 6, 72, "upper"),
                ScoreNote("lower", 0, 12, 48, "lower"),
            ),
        ),
    )
    preview_path = write_neutral_score_preview(
        segments,
        3.0,
        tmp_path / "preview.mid",
        meta_track_name="Test",
        note_track_name="Notes",
    )
    midi = mido.MidiFile(preview_path)
    first_note_on = next(
        message for track in midi.tracks for message in track if message.type == "note_on"
    )
    first_note_on.note += 1
    midi.save(preview_path)

    with pytest.raises(ValueError, match="note events do not match"):
        check_neutral_score_preview(segments, 3.0, preview_path)
