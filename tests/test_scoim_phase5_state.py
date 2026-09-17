import json

import pytest
from test_scoim_phase5_realization import SequencedRunner, _document, _response, _upstream

from scoim.phase5_realization import Phase5Request, realize_phase5
from scoim.phase5_state import load_complete_phase5_state


def _completed_phase5(tmp_path):
    document = _document()
    phase3_state, phase4_state, phase4_ledger = _upstream(tmp_path)
    phase5_dir = tmp_path / "phase5"
    result = realize_phase5(
        Phase5Request(document, phase3_state, phase4_state, phase4_ledger),
        SequencedRunner([_response(), _response(6)]),
        phase5_dir,
    )
    assert result.realized
    phase5_state = json.loads(
        (phase5_dir / "outputs" / "phase5-state.json").read_text(encoding="utf-8")
    )
    cumulative_ledger = json.loads(
        (phase5_dir / "outputs" / "projection-ledger.json").read_text(encoding="utf-8")
    )
    return document, phase3_state, phase4_state, phase5_state, cumulative_ledger


def test_load_complete_phase5_state_reads_a_valid_complete_boundary(tmp_path) -> None:
    document, phase3_state, phase4_state, phase5_state, cumulative_ledger = _completed_phase5(
        tmp_path
    )

    loaded = load_complete_phase5_state(
        document,
        phase3_state,
        phase4_state,
        phase5_state,
        cumulative_ledger,
    )

    assert set(loaded.notes_by_material_placement) == {
        "support-first",
        "support-return",
    }
    assert loaded.phase4.notes_by_material_placement
    assert loaded.cumulative_projection_ledger


def test_load_complete_phase5_state_requires_every_accompaniment_placement(tmp_path) -> None:
    document, phase3_state, phase4_state, phase5_state, cumulative_ledger = _completed_phase5(
        tmp_path
    )
    phase5_state["notes_by_material_placement"].pop("support-return")

    with pytest.raises(ValueError, match="exactly cover accompaniment placements"):
        load_complete_phase5_state(
            document,
            phase3_state,
            phase4_state,
            phase5_state,
            cumulative_ledger,
        )


def test_load_complete_phase5_state_rejects_a_changed_saved_note(tmp_path) -> None:
    document, phase3_state, phase4_state, phase5_state, cumulative_ledger = _completed_phase5(
        tmp_path
    )
    phase5_state["notes_by_material_placement"]["support-first"][0]["pitch"] += 1

    with pytest.raises(ValueError, match="reconstructed accompaniment notes"):
        load_complete_phase5_state(
            document,
            phase3_state,
            phase4_state,
            phase5_state,
            cumulative_ledger,
        )


def test_load_complete_phase5_state_rejects_a_changed_local_ledger(tmp_path) -> None:
    document, phase3_state, phase4_state, phase5_state, cumulative_ledger = _completed_phase5(
        tmp_path
    )
    local_ledger = phase5_state["projection_ledger"]
    local_ledger[0]["evidence"] = "changed evidence"
    cumulative_ledger[-len(local_ledger)]["evidence"] = "changed evidence"

    with pytest.raises(ValueError, match="projection ledger does not match"):
        load_complete_phase5_state(
            document,
            phase3_state,
            phase4_state,
            phase5_state,
            cumulative_ledger,
        )


def test_load_complete_phase5_state_rejects_a_changed_upstream_state(tmp_path) -> None:
    document, phase3_state, phase4_state, phase5_state, cumulative_ledger = _completed_phase5(
        tmp_path
    )
    phase4_state["outcome"] = "changed"

    with pytest.raises(ValueError, match="does not match the phase-4 state"):
        load_complete_phase5_state(
            document,
            phase3_state,
            phase4_state,
            phase5_state,
            cumulative_ledger,
        )
