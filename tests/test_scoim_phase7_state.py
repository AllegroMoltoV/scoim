import json

import pytest
from test_scoim_phase3_realization import SequencedRunner
from test_scoim_phase6_state import _phase6_run
from test_scoim_phase7_realization import _response

from llm_musical_composer.run_state import sha256_json
from scoim.phase7_realization import Phase7Request, realize_phase7
from scoim.phase7_state import load_complete_phase7_run


def _phase7_run(tmp_path):
    phase6_dir = _phase6_run(tmp_path)
    phase7_dir = tmp_path / "phase7"
    result = realize_phase7(Phase7Request(phase6_dir), SequencedRunner([_response()]), phase7_dir)
    assert result.realized
    return phase7_dir


def _rewrite_as_phase7_schema_v1(run_dir) -> None:
    run_spec_path = run_dir / "run-spec.json"
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    run_spec["schema_version"] = 1
    run_spec_path.write_text(json.dumps(run_spec), encoding="utf-8")

    score = json.loads((run_dir / "inputs" / "score-spec.json").read_text(encoding="utf-8"))
    voice_by_note_id = {
        note["score_note_id"]: note["voice"]
        for unit in score["score_units"]
        for layer in unit["score_unit_layers"]
        for note in layer["notes"]
    }
    rendered_path = run_dir / "outputs" / "rendered-performance.json"
    rendered = json.loads(rendered_path.read_text(encoding="utf-8"))
    for note in rendered["notes"]:
        source_ids = note.pop("source_score_note_ids")
        assert len(source_ids) == 1
        note["source_score_note_id"] = source_ids[0]
        note["voice"] = voice_by_note_id[source_ids[0]]
    rendered_path.write_text(json.dumps(rendered), encoding="utf-8")

    state_path = run_dir / "outputs" / "phase7-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["rendered_performance_sha256"] = sha256_json(rendered)
    state_path.write_text(json.dumps(state), encoding="utf-8")


def test_phase7_run_reconstructs_without_calling_a_model(tmp_path) -> None:
    run_dir = _phase7_run(tmp_path)

    loaded = load_complete_phase7_run(run_dir)

    run_spec = json.loads((run_dir / "run-spec.json").read_text(encoding="utf-8"))
    assert run_spec["schema_version"] == 2
    assert loaded.phase7_schema_version == 2
    assert loaded.performance.section_performances[0].section_id == "statement"
    assert loaded.rendered.notes
    assert loaded.cumulative_projection_ledger[-1].status == "unverified"


def test_phase7_schema_v1_run_reconstructs_with_its_saved_contract(tmp_path) -> None:
    run_dir = _phase7_run(tmp_path)
    _rewrite_as_phase7_schema_v1(run_dir)

    loaded = load_complete_phase7_run(run_dir)

    assert loaded.phase7_schema_version == 1
    assert loaded.rendered.notes


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
