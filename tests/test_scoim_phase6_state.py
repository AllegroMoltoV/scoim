import json

import pytest
from test_scoim_phase5_state import _completed_phase5

from scoim.phase6_realization import Phase6Request, realize_phase6
from scoim.phase6_state import load_complete_phase6_run


def test_load_complete_phase6_run_reconstructs_the_saved_score(tmp_path) -> None:
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

    loaded = load_complete_phase6_run(run_dir)

    assert loaded.score.score_units
    assert loaded.phase5.notes_by_material_placement
    assert loaded.cumulative_projection_ledger


def test_load_complete_phase6_run_rejects_a_changed_input_file(tmp_path) -> None:
    run_dir = _phase6_run(tmp_path)
    phase3_path = run_dir / "inputs" / "phase3-state.json"
    phase3_state = json.loads(phase3_path.read_text(encoding="utf-8"))
    phase3_state["outcome"] = "changed"
    phase3_path.write_text(json.dumps(phase3_state), encoding="utf-8")

    with pytest.raises(ValueError, match="input hash does not match"):
        load_complete_phase6_run(run_dir)


def test_load_complete_phase6_run_rejects_a_changed_score_file(tmp_path) -> None:
    run_dir = _phase6_run(tmp_path)
    score_path = run_dir / "outputs" / "score-spec.json"
    score = json.loads(score_path.read_text(encoding="utf-8"))
    score["divisions"] += 1
    score_path.write_text(json.dumps(score), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash does not match"):
        load_complete_phase6_run(run_dir)


def _phase6_run(tmp_path):
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
    return run_dir
