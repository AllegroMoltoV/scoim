import copy
import json
from pathlib import Path

import pytest
from test_scoim_phase4_realization import SequencedRunner, _document, _phase3_inputs
from test_scoim_phase5_realization import _response as _accompaniment

from llm_musical_composer.run_state import RunStore, sha256_file
from scoim.score_realization import ScoreRequest, realize_score
from scoim.score_state import load_complete_score_run
from scoim.score_work_plan import build_score_work_plan


def _responses(document):
    pitches = {
        "theme-first": (72, 12, "upper"),
        "theme-return": (74, 8, "upper"),
        "ending-only": (60, 12, "lower"),
        "bridge-only": (71, 12, "upper"),
    }
    placements = document["script"]["material_placements"]
    result = []
    for operation in build_score_work_plan(document).operations:
        response = {"foregrounds": [], "accompaniments": []}
        for placement_id in operation.material_placement_ids:
            if placements[placement_id]["role"] == "foreground":
                pitch, duration, voice = pitches[placement_id]
                response["foregrounds"].append(
                    {
                        "notes": [
                            {
                                "at_units": 0,
                                "duration_units": duration,
                                "pitch": pitch,
                                "voice": voice,
                            }
                        ]
                    }
                )
            else:
                response["accompaniments"].append(
                    _accompaniment(0 if placement_id == "support-first" else 6)
                )
        result.append(response)
    return result


def _score_run(tmp_path: Path, document=None) -> Path:
    document = document or _document()
    state, ledger = _phase3_inputs(tmp_path, document)
    result = realize_score(
        ScoreRequest(document, state, ledger),
        SequencedRunner(_responses(document)),
        tmp_path / "score",
    )
    assert result.realized, result.issues
    return tmp_path / "score"


def test_score_completes_and_reloads_without_model_calls(tmp_path):
    run = _score_run(tmp_path)
    loaded = load_complete_score_run(run)
    assert set(loaded.notes_by_material_placement) == set(
        _document()["script"]["material_placements"]
    )
    assert (run / "outputs/foreground-preview.mid").is_file()
    assert (run / "outputs/score-preview.mid").is_file()
    request = ScoreRequest(
        _document(),
        json.loads((run / "inputs/phase3-state.json").read_text(encoding="utf-8")),
        json.loads((run / "inputs/projection-ledger.json").read_text(encoding="utf-8")),
    )
    runner = SequencedRunner([])
    assert realize_score(request, runner, run).realized
    assert not runner.prompts


def test_score_resume_reconstructs_progress_from_accepted(tmp_path):
    document = _document()
    state, ledger = _phase3_inputs(tmp_path, document)
    request = ScoreRequest(document, state, ledger)
    responses = _responses(document)
    run = tmp_path / "score"
    result = realize_score(request, SequencedRunner(responses[:1]), run, max_new_operations=1)
    assert result.outcome == "paused"
    (run / "progress/score-state.json").write_text("{}", encoding="utf-8")
    runner = SequencedRunner(responses[1:])
    assert realize_score(request, runner, run).realized
    assert len(runner.prompts) == len(responses) - 1
    load_complete_score_run(run)


def test_score_recovers_interruption_after_acceptance_before_group_snapshot(tmp_path, monkeypatch):
    document = _document()
    state, ledger = _phase3_inputs(tmp_path, document)
    request = ScoreRequest(document, state, ledger)
    responses = _responses(document)
    run = tmp_path / "score"
    original = RunStore.snapshot_json

    def interrupted(self, path, value):
        if str(path).startswith("groups/"):
            raise RuntimeError("interrupted")
        return original(self, path, value)

    with monkeypatch.context() as patch:
        patch.setattr(RunStore, "snapshot_json", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            realize_score(request, SequencedRunner(responses[:1]), run)
    assert not (run / "outputs").exists()
    runner = SequencedRunner(responses[1:])
    assert realize_score(request, runner, run).realized
    assert len(runner.prompts) == len(responses) - 1


def test_score_loader_rejects_changed_comparison_evidence(tmp_path):
    run = _score_run(tmp_path)
    group = next((run / "groups").glob("*.json"))
    value = json.loads(group.read_text(encoding="utf-8"))
    value["prompt_sha256"] = "0" * 64
    group.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="comparison evidence"):
        load_complete_score_run(run)


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_joint_group_repair_never_accepts_only_its_valid_foreground(tmp_path, repair_succeeds):
    document = _document()
    document["script"]["script_element_variation_relations"]["theme-varied"].update(
        source={"type": "section", "id": "bridge"},
        target={"type": "section", "id": "return"},
    )
    operations = build_score_work_plan(document).operations
    index = next(
        i
        for i, operation in enumerate(operations)
        if "support-return" in operation.material_placement_ids
    )
    operation = operations[index]
    assert set(operation.material_placement_ids) == {
        "bridge-only",
        "theme-return",
        "support-return",
    }
    responses = _responses(document)
    invalid = copy.deepcopy(responses[index])
    invalid["accompaniments"][0]["events"][0]["degree"] = "seventh"
    attempts = [invalid]
    attempts += responses[index:] if repair_succeeds else [invalid]
    state, ledger = _phase3_inputs(tmp_path, document)
    run = tmp_path / "score"
    request = ScoreRequest(document, state, ledger)
    paused = realize_score(
        request, SequencedRunner(responses[:index]), run, max_new_operations=index
    )
    assert paused.outcome == "paused"
    upstream = {path: sha256_file(path) for path in (run / "events").rglob("accepted.json")}
    runner = SequencedRunner(attempts)
    result = realize_score(request, runner, run)
    assert all(sha256_file(path) == digest for path, digest in upstream.items())
    if repair_succeeds:
        assert result.realized, result.issues
        loaded = load_complete_score_run(run)
        assert set(operation.material_placement_ids).issubset(loaded.notes_by_material_placement)
        assert len(runner.prompts) == len(operations) - index + 1
    else:
        assert not result.realized
        assert not (run / "events" / operation.operation_id / "accepted.json").exists()
        assert not (run / "groups" / f"{operation.operation_id}.json").exists()
        progress = json.loads((run / "progress/score-state.json").read_text(encoding="utf-8"))
        assert not set(operation.material_placement_ids).intersection(
            progress["notes_by_material_placement"]
        )
        assert not (run / "outputs").exists()


def test_score_recovers_final_output_interruption_without_new_model_calls(tmp_path, monkeypatch):
    document = _document()
    state, ledger = _phase3_inputs(tmp_path, document)
    request = ScoreRequest(document, state, ledger)
    run = tmp_path / "score"

    def interrupt(*args, **kwargs):
        raise RuntimeError("final output interruption")

    with monkeypatch.context() as patch:
        patch.setattr("scoim.score_realization.write_neutral_score_preview", interrupt)
        with pytest.raises(RuntimeError, match="final output interruption"):
            realize_score(request, SequencedRunner(_responses(document)), run)
    assert not (run / "outputs").exists()
    assert len(list((run / "events").glob("*/accepted.json"))) == len(_responses(document))
    runner = SequencedRunner([])
    assert realize_score(request, runner, run).realized
    assert not runner.prompts
    load_complete_score_run(run)
