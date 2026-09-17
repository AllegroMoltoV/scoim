import json

import mido
from test_scoim_phase5_state import _completed_phase5

import scoim.phase6_realization as phase6_realization
from scoim.phase6_realization import Phase6Request, realize_phase6


def test_phase6_preserves_every_validated_note_in_one_score_spec(tmp_path) -> None:
    document, phase3_state, phase4_state, phase5_state, cumulative_ledger = _completed_phase5(
        tmp_path
    )
    run_dir = tmp_path / "phase6"

    result = realize_phase6(
        Phase6Request(
            document,
            phase3_state,
            phase4_state,
            phase5_state,
            cumulative_ledger,
        ),
        run_dir,
    )

    assert result.realized
    assert result.outcome == "complete"
    score = json.loads((run_dir / "outputs" / "score-spec.json").read_text(encoding="utf-8"))
    actual_notes = {
        layer["source_material_placement_id"]: layer["notes"]
        for unit in score["score_units"]
        for layer in unit["score_unit_layers"]
    }
    expected_notes = {
        **phase4_state["notes_by_material_placement"],
        **phase5_state["notes_by_material_placement"],
    }
    assert actual_notes == expected_notes
    assert all(unit["directions"] == [] for unit in score["score_units"])


def test_phase6_saves_a_neutral_preview_with_every_score_note(tmp_path) -> None:
    document, phase3_state, phase4_state, phase5_state, cumulative_ledger = _completed_phase5(
        tmp_path
    )
    run_dir = tmp_path / "phase6"

    result = realize_phase6(
        Phase6Request(
            document,
            phase3_state,
            phase4_state,
            phase5_state,
            cumulative_ledger,
        ),
        run_dir,
    )

    assert result.realized
    midi = mido.MidiFile(run_dir / "outputs" / "score-preview.mid")
    messages = [message for track in midi.tracks for message in track]
    note_count = sum(
        len(notes)
        for notes in (
            *phase4_state["notes_by_material_placement"].values(),
            *phase5_state["notes_by_material_placement"].values(),
        )
    )
    assert midi.length == 180.0
    assert sum(message.type == "note_on" for message in messages) == note_count
    assert sum(message.type == "note_off" for message in messages) == note_count
    assert {message.channel for message in messages if message.type == "note_on"} == {0, 1}
    assert all(message.velocity == 64 for message in messages if message.type == "note_on")
    assert not any(message.type == "control_change" for message in messages)


def test_phase6_does_not_publish_a_run_when_preview_verification_fails(
    tmp_path, monkeypatch
) -> None:
    document, phase3_state, phase4_state, phase5_state, cumulative_ledger = _completed_phase5(
        tmp_path
    )
    run_dir = tmp_path / "phase6"

    def fail_preview(*_args, **_kwargs) -> None:
        raise ValueError("changed preview")

    monkeypatch.setattr(
        phase6_realization,
        "check_neutral_score_preview",
        fail_preview,
        raising=False,
    )

    result = realize_phase6(
        Phase6Request(
            document,
            phase3_state,
            phase4_state,
            phase5_state,
            cumulative_ledger,
        ),
        run_dir,
    )

    assert not result.realized
    assert result.outcome == "preview_invalid"
    assert not run_dir.exists()
    assert result.run_dir is not None and result.run_dir.exists()


def test_phase6_does_not_publish_a_run_when_saved_state_cannot_be_reloaded(
    tmp_path, monkeypatch
) -> None:
    document, phase3_state, phase4_state, phase5_state, cumulative_ledger = _completed_phase5(
        tmp_path
    )
    run_dir = tmp_path / "phase6"

    def fail_reload(*_args, **_kwargs) -> None:
        raise ValueError("changed saved state")

    monkeypatch.setattr(
        phase6_realization,
        "load_complete_phase6_run",
        fail_reload,
        raising=False,
    )

    result = realize_phase6(
        Phase6Request(
            document,
            phase3_state,
            phase4_state,
            phase5_state,
            cumulative_ledger,
        ),
        run_dir,
    )

    assert not result.realized
    assert result.outcome == "persisted_state_invalid"
    assert not run_dir.exists()
    assert result.run_dir is not None and result.run_dir.exists()


def test_phase6_records_an_invalid_upstream_request_without_publishing_it(
    tmp_path,
) -> None:
    document, phase3_state, phase4_state, phase5_state, cumulative_ledger = _completed_phase5(
        tmp_path
    )
    phase5_state["outcome"] = "changed"
    run_dir = tmp_path / "phase6"

    result = realize_phase6(
        Phase6Request(
            document,
            phase3_state,
            phase4_state,
            phase5_state,
            cumulative_ledger,
        ),
        run_dir,
    )

    assert not result.realized
    assert result.outcome == "request_invalid"
    assert not run_dir.exists()
    assert result.run_dir is not None and result.run_dir.exists()
    failure = json.loads(
        (result.run_dir / "outputs" / "realization.json").read_text(encoding="utf-8")
    )
    assert failure["outcome"] == "request_invalid"
    assert failure["issues"][0]["path"] == "/phase6_request"
