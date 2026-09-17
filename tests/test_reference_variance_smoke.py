import json
from pathlib import Path

import pytest
from test_staged_pipeline_generation import FakeRunner, _responses

from llm_musical_composer.reference_variance_smoke import (
    DEFAULT_ARTIFACT_ROOT,
    SMOKE_VERSION,
    execute_prepared_smoke,
    main,
    prepare_reference_variance_smoke,
)
from llm_musical_composer.run_state import StateConflictError

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_fixes_four_runs_and_all_input_hashes(tmp_path: Path) -> None:
    result = prepare_reference_variance_smoke(ROOT, tmp_path / "smoke")

    assert result["status"] == "prepared"
    assert result["selection_sha256"] == (
        "e2388928f3555de4283a91b51a9d7c9000962ee85a065c21a32c5f4f2f828180"
    )
    assert [item["run_id"] for item in result["runs"]] == [
        "reference-a-candidate-1",
        "reference-a-candidate-2",
        "reference-b-candidate-1",
        "reference-b-candidate-2",
    ]
    assert result["maximum_external_call_count"] == 28
    assert result["smoke_version"] == "reference-variance-smoke-v8"
    assert SMOKE_VERSION == "reference-variance-smoke-v8"
    assert Path(".appendix/reference-variance-smoke-v8") == DEFAULT_ARTIFACT_ROOT
    assert result["model_config"] == {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "timeout_seconds": 600,
    }
    assert all(len(item["prompt_target_sha256"]) == 64 for item in result["runs"])
    assert {
        "orchestrator",
        "quality",
        "recurrence_analysis",
        "recurrence_quality",
        "run_state",
        "smf_notes",
        "reference_variance_smoke",
        "prompt_piece_plan",
        "prompt_score_spec",
        "prompt_performance_spec",
    }.issubset(result["implementation_hashes"])
    assert (tmp_path / "smoke" / "preflight.json").is_file()


def test_prepare_does_not_overwrite_changed_preflight(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    prepare_reference_variance_smoke(ROOT, root)
    path = root / "preflight.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["model_config"]["model"] = "changed"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(StateConflictError, match="preflight"):
        prepare_reference_variance_smoke(ROOT, root)


def test_execute_runs_only_the_selected_candidate_and_stage(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    prepare_reference_variance_smoke(ROOT, root)
    runners = []

    def factory(run_dir, model_config):
        assert run_dir.name == "reference-b-candidate-2"
        assert model_config["timeout_seconds"] == 600
        runner = FakeRunner(_responses())
        runners.append(runner)
        return runner

    result = execute_prepared_smoke(
        ROOT,
        root,
        run_id="reference-b-candidate-2",
        stop_after="piece_plan",
        runner_factory=factory,
    )

    assert len(runners) == 1
    assert [call[0] for call in runners[0].calls] == ["piece-plan"]
    assert result.status == "stopped"
    assert result.stopped_after == "piece_plan"


def test_execute_rejects_an_unknown_run_before_constructing_runner(tmp_path: Path) -> None:
    root = tmp_path / "smoke"
    prepare_reference_variance_smoke(ROOT, root)

    with pytest.raises(ValueError, match="unknown prepared run"):
        execute_prepared_smoke(
            ROOT,
            root,
            run_id="not-prepared",
            stop_after="piece_plan",
            runner_factory=lambda *args: pytest.fail("runner must not be constructed"),
        )


def test_execute_rejects_an_implementation_change_before_constructing_runner(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "smoke"
    prepare_reference_variance_smoke(ROOT, root)
    monkeypatch.setattr(
        "llm_musical_composer.reference_variance_smoke.staged_pipeline_fingerprints",
        lambda: {"changed": "f" * 64},
    )

    with pytest.raises(StateConflictError, match="preflight"):
        execute_prepared_smoke(
            ROOT,
            root,
            run_id="reference-a-candidate-1",
            stop_after="piece_plan",
            runner_factory=lambda *args: pytest.fail("runner must not be constructed"),
        )


def test_execute_cli_requires_run_and_stop_stage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "reference-variance-smoke",
            "execute",
            "--project-root",
            str(ROOT),
            "--artifact-root",
            str(tmp_path / "smoke"),
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
