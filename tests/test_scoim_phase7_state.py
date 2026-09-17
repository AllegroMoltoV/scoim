import json

import pytest
from test_scoim_phase3_realization import SequencedRunner
from test_scoim_phase6_state import _phase6_run
from test_scoim_phase7_realization import _response

from scoim.phase7_realization import Phase7Request, realize_phase7
from scoim.phase7_state import load_complete_phase7_run


def _phase7_run(tmp_path):
    phase6_dir = _phase6_run(tmp_path)
    phase7_dir = tmp_path / "phase7"
    result = realize_phase7(Phase7Request(phase6_dir), SequencedRunner([_response()]), phase7_dir)
    assert result.realized
    return phase7_dir


def test_phase7_run_reconstructs_without_calling_a_model(tmp_path) -> None:
    run_dir = _phase7_run(tmp_path)

    loaded = load_complete_phase7_run(run_dir)

    assert loaded.performance.section_performances[0].section_id == "statement"
    assert loaded.rendered.notes
    assert loaded.cumulative_projection_ledger[-1].status == "unverified"


def test_phase7_run_rejects_a_changed_saved_performance(tmp_path) -> None:
    run_dir = _phase7_run(tmp_path)
    path = run_dir / "outputs" / "performance-spec.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["default_velocity"] = 65
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="saved phase-7 performance"):
        load_complete_phase7_run(run_dir)


def test_phase7_run_rejects_an_unverifiable_model_operation_record(tmp_path) -> None:
    run_dir = _phase7_run(tmp_path)
    validation_path = next((run_dir / "attempts").glob("*/attempt-*/validation.json"))
    validation_path.unlink()

    with pytest.raises(ValueError, match="model operation record"):
        load_complete_phase7_run(run_dir)
