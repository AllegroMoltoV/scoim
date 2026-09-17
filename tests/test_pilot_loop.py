from __future__ import annotations

import json
import subprocess
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_musical_composer.material_development import evaluate_material_development
from llm_musical_composer.music_dsl import parse_composition
from llm_musical_composer.pilot_features import build_reference_profile, extract_features
from llm_musical_composer.pilot_loop import (
    CodexExecRunner,
    CodexProcessRecoveryError,
    RunStatus,
    _compose_prompt,
    _evaluate_source,
    _repair_prompt,
    _resolve_codex_executable,
    _response_source,
    _revision_improves,
    _run_codex_process,
    isolated_codex_working_directory,
    main,
    run_pilot,
    validate_revision_scope,
)
from llm_musical_composer.run_state import (
    InterruptedAttemptError,
    RunStore,
    StateConflictError,
)
from llm_musical_composer.section_contrast import evaluate_section_contrast
from llm_musical_composer.sustain_profile import evaluate_sustain_profile
from tests.test_music_dsl import (
    CONTRACT_ENDING_SOURCE,
    CONTRACT_SOURCE,
    ENDING_SOURCE,
    VALID_SOURCE,
)
from tests.test_pilot_features import notes


class FakeRunner:
    def __init__(self, sources: list[str]) -> None:
        self.sources = deque(sources)
        self.calls: list[str] = []

    def run(
        self, step_id: str, prompt: str, input_hashes: dict[str, str] | None = None
    ) -> dict[str, object]:
        del step_id, input_hashes
        self.calls.append(prompt)
        source = self.sources.popleft()
        response = {"composition_source": source, "intent_summary": "test"}
        return response


def test_revision_scope_accepts_only_target_material_change() -> None:
    before = parse_composition(VALID_SOURCE)
    after = parse_composition(VALID_SOURCE.replace("pitch=60", "pitch=61"))

    assert validate_revision_scope(before, after, "A") == []
    assert validate_revision_scope(before, after, "B") == [
        "material A changed outside target",
        "target material did not change",
    ]

    renamed = replace(after, title="renamed")
    changed_form = replace(after, form=(after.form[0], after.form[2], after.form[1]))
    fewer_materials = replace(after, materials=after.materials[:1])
    assert "title changed" in validate_revision_scope(before, renamed, "A")
    assert "form changed" in validate_revision_scope(before, changed_form, "A")
    assert validate_revision_scope(before, fewer_materials, "A") == ["material set changed"]


def _evaluation(distance: float, *, inside: int = 0) -> dict[str, object]:
    return {
        "inside_axis_count": inside,
        "worst_axis_distance": distance,
        "axes": {"pitch_order": {"normalized_distance": distance}},
    }


def _development_profile() -> dict[str, object]:
    return {
        "metrics": {
            "duration_change_rate": {"p25": 0.2, "median": 0.5, "p75": 0.8},
            "ioi_change_rate": {"p25": 0.2, "median": 0.5, "p75": 0.8},
            "attack_size_change_rate": {"p25": 0.2, "median": 0.5, "p75": 0.8},
        }
    }


def _development_report(count: int, worst: str = "A") -> dict[str, object]:
    return {
        "status": "pass",
        "extreme_material_count": count,
        "worst_material_id": worst,
        "materials": [],
    }


def test_run_pilot_generates_three_candidates_and_accepts_improvement(
    tmp_path, monkeypatch
) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    revised = CONTRACT_ENDING_SOURCE.replace("pitch=72", "pitch=73", 1)
    runner = FakeRunner([CONTRACT_ENDING_SOURCE] * 3 + [revised])
    evaluations = iter([_evaluation(2.0), _evaluation(3.0), _evaluation(4.0), _evaluation(1.0)])
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.evaluate_features",
        lambda *args: next(evaluations),
    )

    result = run_pilot(profile, runner, tmp_path / "run", random_seed=5)

    assert result.status is RunStatus.COMPLETED
    assert result.call_count == 4
    assert len(result.candidates) == 3
    assert result.final_midi_path == tmp_path / "run" / "final.mid"
    assert result.final_source_path == tmp_path / "run" / "final.music.py"
    assert result.revision_status == "accepted"
    assert result.final_midi_path.is_file()
    assert result.final_source_path.read_text(encoding="utf-8").strip() == revised
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["call_count"] == 4
    assert manifest["selected_candidate"] == "candidate-1"
    assert manifest["revision_status"] == "accepted"
    assert manifest["final_midi_path"].endswith("final.mid")


def test_run_pilot_prefers_fewer_extreme_materials_before_existing_axes(
    tmp_path, monkeypatch
) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    runner = FakeRunner([CONTRACT_ENDING_SOURCE] * 3)
    evaluations = iter([_evaluation(0.0, inside=1) for _ in range(3)])
    development = iter([_development_report(1), _development_report(0), _development_report(1)])
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.evaluate_features",
        lambda *args: next(evaluations),
    )
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.evaluate_material_development",
        lambda *args: next(development),
    )

    result = run_pilot(
        profile,
        runner,
        tmp_path / "run",
        material_development_profile=_development_profile(),
    )

    assert result.selected_candidate == "candidate-2"
    assert result.call_count == 3


def test_run_pilot_revises_the_worst_extreme_material_once(tmp_path, monkeypatch) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    revised = CONTRACT_ENDING_SOURCE.replace("pitch=48", "pitch=49", 1)
    runner = FakeRunner([CONTRACT_ENDING_SOURCE] * 3 + [revised])
    evaluations = iter([_evaluation(0.0, inside=1) for _ in range(4)])
    development = iter(
        [
            _development_report(1),
            _development_report(2),
            _development_report(2),
            _development_report(0),
        ]
    )
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.evaluate_features",
        lambda *args: next(evaluations),
    )
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.evaluate_material_development",
        lambda *args: next(development),
    )

    result = run_pilot(
        profile,
        runner,
        tmp_path / "run",
        material_development_profile=_development_profile(),
    )

    assert result.revision_status == "accepted"
    assert result.call_count == 4
    report = json.loads((tmp_path / "run" / "evaluations.json").read_text(encoding="utf-8"))
    assert report["target_axis"] == "material_development"
    assert report["target_material"] == "A"


def test_run_pilot_rejects_revision_without_automatic_improvement(tmp_path, monkeypatch) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    revised = CONTRACT_ENDING_SOURCE.replace("pitch=72", "pitch=73", 1)
    runner = FakeRunner([CONTRACT_ENDING_SOURCE] * 3 + [revised])
    evaluations = iter([_evaluation(1.0), _evaluation(2.0), _evaluation(3.0), _evaluation(1.5)])
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.evaluate_features",
        lambda *args: next(evaluations),
    )

    result = run_pilot(profile, runner, tmp_path / "run")

    assert result.status is RunStatus.COMPLETED
    assert result.revision_status == "rejected_no_improvement"
    assert result.final_source_path is not None
    assert result.final_source_path.read_text(encoding="utf-8").strip() == CONTRACT_ENDING_SOURCE


def test_run_pilot_skips_revision_when_selected_candidate_has_no_outlier(
    tmp_path, monkeypatch
) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    runner = FakeRunner([CONTRACT_ENDING_SOURCE] * 3)
    evaluations = iter([_evaluation(0.0, inside=1)] * 3)
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.evaluate_features",
        lambda *args: next(evaluations),
    )

    result = run_pilot(profile, runner, tmp_path / "run")

    assert result.status is RunStatus.COMPLETED
    assert result.call_count == 3
    assert result.revision_status == "skipped_no_outlier"
    assert len(runner.calls) == 3
    evaluations_report = json.loads(
        (tmp_path / "run" / "evaluations.json").read_text(encoding="utf-8")
    )
    assert evaluations_report["revision_after"] is None
    assert evaluations_report["target_axis"] == "pitch_order"


def test_run_pilot_does_not_revise_activity_only_outlier(tmp_path, monkeypatch) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    runner = FakeRunner([CONTRACT_ENDING_SOURCE] * 3)
    activity_only = {
        "inside_axis_count": 1,
        "worst_axis_distance": 2.0,
        "axes": {
            "texture": {"inside_iqr": True, "normalized_distance": 0.0},
            "activity": {"inside_iqr": False, "normalized_distance": 2.0},
        },
    }
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.evaluate_features",
        lambda *args: activity_only,
    )

    result = run_pilot(profile, runner, tmp_path / "run")

    assert result.status is RunStatus.COMPLETED
    assert result.call_count == 3
    assert result.revision_status == "skipped_no_revisable_axis"
    assert len(runner.calls) == 3


def test_run_pilot_records_durable_steps_and_call_categories(tmp_path, monkeypatch) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    runner = FakeRunner([CONTRACT_ENDING_SOURCE] * 3)
    evaluations = iter([_evaluation(0.0, inside=1)] * 3)
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.evaluate_features",
        lambda *args: next(evaluations),
    )
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1, "requested_model": "test-model"})

    result = run_pilot(profile, runner, store.run_dir, run_store=store)

    assert result.status is RunStatus.COMPLETED
    state = store.rebuild_state()
    assert state["status"] == "completed"
    assert state["steps"]["repair-candidate-1"]["status"] == "skipped"
    assert state["steps"]["revise-selected"]["status"] == "skipped"
    manifest = json.loads((store.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["call_count"] == 0
    assert manifest["call_count_definition"].startswith("reserved external")
    assert manifest["requested_model"] == "test-model"


def test_run_pilot_records_no_valid_candidate_as_failed_run(tmp_path) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    runner = FakeRunner(["invalid()"] * 6)
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})

    result = run_pilot(profile, runner, store.run_dir, run_store=store)

    assert result.status is RunStatus.NO_VALID_CANDIDATE
    state = store.rebuild_state()
    assert state["status"] == "failed"
    assert state["steps"]["publish-final"]["status"] == "failed"


def test_run_pilot_repairs_invalid_candidates_but_stops_at_call_cap(tmp_path) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    runner = FakeRunner(["invalid()"] * 7)

    result = run_pilot(profile, runner, tmp_path / "run", random_seed=5)

    assert result.status is RunStatus.NO_VALID_CANDIDATE
    assert result.call_count == 6
    assert result.final_midi_path is None


def test_run_pilot_repairs_only_pedals_for_sustain_contract(tmp_path, monkeypatch) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    invalid = CONTRACT_ENDING_SOURCE.replace("value=127", "value=48")
    runner = FakeRunner([invalid, CONTRACT_ENDING_SOURCE] * 3)
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.evaluate_features",
        lambda *args: _evaluation(0.0, inside=1),
    )

    result = run_pilot(profile, runner, tmp_path / "run")

    assert result.status is RunStatus.COMPLETED
    assert result.call_count == 6
    assert all(candidate.repaired for candidate in result.candidates)
    assert all(candidate.error is None for candidate in result.candidates)


def test_run_pilot_rejects_non_pedal_change_during_sustain_repair(tmp_path, monkeypatch) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    invalid = CONTRACT_ENDING_SOURCE.replace("value=127", "value=48")
    changed_note = CONTRACT_ENDING_SOURCE.replace("pitch=48", "pitch=49", 1)
    runner = FakeRunner([invalid, changed_note, CONTRACT_ENDING_SOURCE, CONTRACT_ENDING_SOURCE])
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.evaluate_features",
        lambda *args: _evaluation(0.0, inside=1),
    )

    result = run_pilot(profile, runner, tmp_path / "run")

    assert result.status is RunStatus.COMPLETED
    assert result.call_count == 4
    assert result.candidates[0].error == ("SustainRepairScopeError: material A notes changed")
    assert result.selected_candidate == "candidate-2"


@pytest.mark.parametrize("revision", ["invalid()", CONTRACT_ENDING_SOURCE])
def test_run_pilot_keeps_baseline_when_revision_is_invalid(tmp_path, revision: str) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    runner = FakeRunner([CONTRACT_ENDING_SOURCE] * 3 + [revision])

    result = run_pilot(profile, runner, tmp_path / "run")

    assert result.status is RunStatus.COMPLETED
    assert result.revision_status in {"rejected_invalid", "rejected_scope"}
    assert result.final_midi_path is not None
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["revision_error"]


def test_new_loop_rejects_candidates_without_tonic_ending_contract(tmp_path) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    runner = FakeRunner([CONTRACT_SOURCE] * 6)

    result = run_pilot(profile, runner, tmp_path / "run")

    assert result.status is RunStatus.NO_VALID_CANDIDATE
    assert all(
        "tonic ending contract" in (candidate.error or "") for candidate in result.candidates
    )


def test_candidate_render_is_not_promoted_when_ending_reread_fails(tmp_path, monkeypatch) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._validate_rendered_ending",
        lambda *args: (_ for _ in ()).throw(ValueError("ending mismatch")),
    )

    candidate = _evaluate_source(
        "candidate-1",
        CONTRACT_ENDING_SOURCE,
        tmp_path / "candidates",
        profile,
        repaired=False,
    )

    assert candidate.error == "ValueError: ending mismatch"
    assert not (tmp_path / "candidates" / "candidate-1.music.py").exists()
    assert not (tmp_path / "candidates" / "candidate-1.mid").exists()


def test_candidate_requires_section_contract(tmp_path) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)

    candidate = _evaluate_source(
        "candidate-1", ENDING_SOURCE, tmp_path / "candidates", profile, repaired=False
    )

    assert "section contract is required" in (candidate.error or "")
    assert candidate.evaluation is None


def test_candidate_requires_pilot_duration(tmp_path) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    short = CONTRACT_ENDING_SOURCE.replace("duration_ms=14000", "duration_ms=10000").replace(
        "at_ms=13700", "at_ms=9700"
    )

    candidate = _evaluate_source(
        "candidate-1", short, tmp_path / "candidates", profile, repaired=False
    )

    assert "between 45000 and 60000" in (candidate.error or "")
    assert candidate.evaluation is None


def test_candidate_records_passing_section_contrast(tmp_path) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)

    candidate = _evaluate_source(
        "candidate-1",
        CONTRACT_ENDING_SOURCE,
        tmp_path / "candidates",
        profile,
        repaired=False,
    )

    assert candidate.error is None
    assert candidate.evaluation is not None
    assert candidate.evaluation["section_contrast"]["status"] == "pass"


def test_candidate_records_passing_sustain_profile(tmp_path) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)

    candidate = _evaluate_source(
        "candidate-1",
        CONTRACT_ENDING_SOURCE,
        tmp_path / "candidates",
        profile,
        repaired=False,
    )

    assert candidate.error is None
    assert candidate.evaluation is not None
    assert candidate.evaluation["sustain_profile"]["status"] == "pass"
    assert all(
        material["pedal_on_ratio"] >= 0.85
        for material in candidate.evaluation["sustain_profile"]["materials"]
    )


def test_candidate_rejects_subthreshold_pedal_values_before_render(tmp_path) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    invalid = CONTRACT_ENDING_SOURCE.replace("value=127", "value=48")

    candidate = _evaluate_source(
        "candidate-1", invalid, tmp_path / "candidates", profile, repaired=False
    )

    assert "SustainContractError" in (candidate.error or "")
    assert "0 or 127" in (candidate.error or "")
    assert not (tmp_path / "candidates" / "candidate-1.music.py").exists()
    assert not (tmp_path / "candidates" / "candidate-1.mid").exists()


def test_sustain_only_repair_rejects_note_changes_before_render(tmp_path) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    invalid = CONTRACT_ENDING_SOURCE.replace("value=127", "value=48")
    changed_note = CONTRACT_ENDING_SOURCE.replace("pitch=48", "pitch=49", 1)

    candidate = _evaluate_source(
        "candidate-1",
        changed_note,
        tmp_path / "candidates",
        profile,
        repaired=True,
        sustain_repair_baseline=parse_composition(invalid, require_section_contract=True),
    )

    assert candidate.error == "SustainRepairScopeError: material A notes changed"
    assert not (tmp_path / "candidates" / "candidate-1.music.py").exists()
    assert not (tmp_path / "candidates" / "candidate-1.mid").exists()


def test_sustain_only_repair_accepts_pedal_changes(tmp_path) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)
    invalid = CONTRACT_ENDING_SOURCE.replace("value=127", "value=48")

    candidate = _evaluate_source(
        "candidate-1",
        CONTRACT_ENDING_SOURCE,
        tmp_path / "candidates",
        profile,
        repaired=True,
        sustain_repair_baseline=parse_composition(invalid, require_section_contract=True),
    )

    assert candidate.error is None
    assert candidate.midi_path is not None


def test_candidate_records_material_development_without_rejecting_simple_material(
    tmp_path,
) -> None:
    profile = build_reference_profile([extract_features(notes())], source_count=1)

    candidate = _evaluate_source(
        "candidate-1",
        CONTRACT_ENDING_SOURCE,
        tmp_path / "candidates",
        profile,
        material_development_profile=_development_profile(),
        repaired=False,
    )

    assert candidate.error is None
    assert candidate.evaluation is not None
    assert candidate.evaluation["material_development"]["extreme_material_count"] == 2


def test_revision_must_improve_target_and_not_worsen_selection_order() -> None:
    before = _evaluation(2.0, inside=1)

    assert _revision_improves(before, _evaluation(1.0, inside=1), "pitch_order")
    assert not _revision_improves(before, _evaluation(2.0, inside=1), "pitch_order")
    assert not _revision_improves(before, _evaluation(3.0, inside=2), "pitch_order")


def test_codex_runner_uses_attempt_staging_and_reuses_verified_response(
    tmp_path, monkeypatch
) -> None:
    response = {"composition_source": VALID_SOURCE, "intent_summary": "test"}
    isolated = tmp_path / "isolated"

    def fake_run(command, **kwargs):
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text(json.dumps(response), encoding="utf-8")
        kwargs["stdout"].write(
            json.dumps({"type": "thread.started", "thread_id": "thread-1"})
            + "\n"
            + json.dumps({"type": "error", "message": "recoverable warning"})
            + "\n"
            + json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 12, "output_tokens": 7},
                }
            )
            + "\n"
        )
        assert kwargs["input"] == "prompt"
        assert kwargs["encoding"] == "utf-8"
        assert command[-1] == "-"
        assert "--sandbox" in command and "read-only" in command
        assert "--ignore-user-config" in command
        assert "--skip-git-repo-check" in command
        assert command[command.index("-c") + 1] == 'model_reasoning_effort="low"'
        assert kwargs["timeout"] == 600
        assert command[command.index("-C") + 1] == str(isolated.resolve())
        assert Path(command[command.index("--output-schema") + 1]).is_absolute()
        assert output_path.name == "response.staged.json"
        assert "attempts" in output_path.parts
        assert kwargs["cwd"] == isolated.resolve()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("llm_musical_composer.pilot_loop._run_codex_process", fake_run)
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._resolve_codex_executable",
        lambda: (Path("codex.exe"), None),
    )
    monkeypatch.chdir(tmp_path)
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})
    runner = CodexExecRunner(
        run_store=store,
        schema_path=Path("schema.json"),
        working_directory=isolated,
        reasoning_effort="low",
        timeout_seconds=600,
    )

    Path("schema.json").write_text(
        '{"type":"object","required":["composition_source","intent_summary"]}',
        encoding="utf-8",
    )
    assert runner.run("compose-1", "prompt") == response
    assert (tmp_path / "run" / "responses" / "compose-1.json").is_file()
    assert runner.run("compose-1", "prompt") == response
    assert runner.call_number == 1
    state = store.rebuild_state()
    assert state["calls"]["saved_response_count"] == 1
    assert state["calls"]["confirmed_external_call_count"] == 1
    terminal = json.loads(
        next((tmp_path / "run" / "attempts" / "compose-1").glob("*/terminal.json")).read_text(
            encoding="utf-8"
        )
    )
    assert terminal["detail"] == "completed with 1 warning error events"


def test_codex_runner_records_a_timeout_as_interrupted(tmp_path, monkeypatch) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._resolve_codex_executable",
        lambda: (Path("codex.exe"), None),
    )
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._run_codex_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("codex", kwargs["timeout"])
        ),
    )
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})
    runner = CodexExecRunner(
        run_store=store,
        schema_path=schema,
        reasoning_effort="low",
        timeout_seconds=600,
    )

    with pytest.raises(InterruptedAttemptError, match="timed out"):
        runner.run("compose-1", "prompt")

    attempt = next((tmp_path / "run" / "attempts" / "compose-1").glob("attempt-*"))
    terminal = json.loads((attempt / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "interrupted"
    assert terminal["detail"] == "codex exec timed out after 600 seconds"


def test_windows_codex_resolver_uses_the_managed_native_executable(
    tmp_path, monkeypatch
) -> None:
    package_root = tmp_path / "node_modules" / "@openai" / "codex"
    executable = (
        package_root
        / "node_modules"
        / "@openai"
        / "codex-win32-x64"
        / "vendor"
        / "x86_64-pc-windows-msvc"
        / "bin"
        / "codex.exe"
    )
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setenv("CODEX_MANAGED_PACKAGE_ROOT", str(package_root))

    resolved, managed_root = _resolve_codex_executable()

    assert resolved == executable.resolve()
    assert managed_root == package_root.resolve()


def test_codex_process_timeout_kills_and_reaps_the_direct_process(
    tmp_path, monkeypatch
) -> None:
    class FakeProcess:
        returncode = None

        def __init__(self) -> None:
            self.communicate_count = 0
            self.killed = False

        def communicate(self, input=None, timeout=None):
            self.communicate_count += 1
            if self.communicate_count == 1:
                raise subprocess.TimeoutExpired("codex", timeout)
            self.returncode = 1
            return (None, None)

        def kill(self) -> None:
            self.killed = True

    process = FakeProcess()
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    stdout_path = tmp_path / "stdout.jsonl"
    stderr_path = tmp_path / "stderr.log"

    with (
        stdout_path.open("w", encoding="utf-8") as stdout_file,
        stderr_path.open("w", encoding="utf-8") as stderr_file,
        pytest.raises(subprocess.TimeoutExpired),
    ):
        _run_codex_process(
            ["codex.exe", "exec"],
            input="prompt",
            text=True,
            encoding="utf-8",
            stdout=stdout_file,
            stderr=stderr_file,
            cwd=None,
            timeout=1,
            env={},
        )

    assert process.killed
    assert process.communicate_count == 2


def test_codex_runner_records_recovery_required_when_process_cannot_be_reaped(
    tmp_path, monkeypatch
) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._resolve_codex_executable",
        lambda: (Path("codex.exe"), None),
    )
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._run_codex_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            CodexProcessRecoveryError("kill failed")
        ),
    )
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})
    runner = CodexExecRunner(run_store=store, schema_path=schema, timeout_seconds=1)

    with pytest.raises(InterruptedAttemptError, match="recovery is required"):
        runner.run("compose-1", "prompt")

    attempt = next((tmp_path / "run" / "attempts" / "compose-1").glob("attempt-*"))
    assert not (attempt / "terminal.json").exists()
    assert json.loads((attempt / "recovery.json").read_text(encoding="utf-8")) == {
        "status": "recovery_required",
        "detail": "kill failed",
    }
    assert store.rebuild_state()["status"] == "recovery_required"
    with pytest.raises(InterruptedAttemptError, match="recovery is required"):
        runner.run("compose-1", "prompt")
    assert runner.call_number == 1


def test_codex_runner_rejects_invalid_execution_settings(tmp_path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    store = RunStore(tmp_path / "run")
    with pytest.raises(ValueError, match="reasoning effort"):
        CodexExecRunner(run_store=store, schema_path=schema, reasoning_effort="extreme")
    with pytest.raises(ValueError, match="timeout"):
        CodexExecRunner(run_store=store, schema_path=schema, timeout_seconds=0)


def test_isolated_working_directory_is_reusable_and_empty(tmp_path) -> None:
    first = isolated_codex_working_directory(tmp_path)
    second = isolated_codex_working_directory(tmp_path)

    assert first == second == (tmp_path / "llm-musical-composer-codex-compose").resolve()
    assert first.is_dir()
    assert list(first.iterdir()) == []


def test_generation_and_repair_prompts_contain_complete_dsl_contract() -> None:
    compose = _compose_prompt({"axes": {}}, 1)
    repair = _repair_prompt("invalid()", "invalid syntax")

    for prompt in (compose, repair):
        assert "composition(" in prompt
        assert "ending=tonic_hold(duration_ms=4000)" in prompt
        assert 'role="opening"' in prompt
        assert 'attack_style="' not in prompt
        assert "id、events、start_ms" in prompt
        assert "Beads" in prompt
        assert "0 から 63" in prompt
        assert "64 から 127" in prompt
        assert "value=127" in prompt
        example = prompt.split("```python\n", 1)[1].split("\n```", 1)[0]
        parsed = parse_composition(example, require_section_contract=True)
        assert parsed.body_duration_ms == 42000
        assert parsed.material_by_id["A"].notes[0].event_id == "a1"
        assert parsed.material_by_id["A"].pedals[0].event_id == "p1"
        assert evaluate_section_contrast(parsed)["status"] == "pass"
        assert evaluate_sustain_profile(parsed.materials)["status"] == "pass"
        assert (
            evaluate_material_development(parsed.materials, _development_profile())[
                "extreme_material_count"
            ]
            == 0
        )


def test_codex_runner_reports_command_failure_and_call_limit(tmp_path, monkeypatch) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._resolve_codex_executable",
        lambda: (Path("codex.exe"), None),
    )
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._run_codex_process",
        lambda *args, **kwargs: SimpleNamespace(returncode=3),
    )
    store = RunStore(tmp_path / "run", max_calls=1)
    store.initialize({"schema_version": 1})
    runner = CodexExecRunner(run_store=store, schema_path=schema)
    with pytest.raises(RuntimeError, match="codex exec failed"):
        runner.run("compose-1", "prompt")
    with pytest.raises(RuntimeError, match="limit"):
        runner.run("compose-2", "prompt")


def test_codex_runner_reports_missing_command(tmp_path, monkeypatch) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("llm_musical_composer.pilot_loop.shutil.which", lambda name: None)
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})
    runner = CodexExecRunner(run_store=store, schema_path=schema)

    with pytest.raises(FileNotFoundError, match="PATH"):
        runner.run("compose-1", "prompt")

    calls = store.rebuild_state()["calls"]
    assert calls["preflight_failure_count"] == 1
    assert calls["failed_external_call_count"] == 0


def test_codex_runner_records_process_start_failure_as_preflight_failure(
    tmp_path, monkeypatch
) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._resolve_codex_executable",
        lambda: (Path("codex.exe"), None),
    )
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._run_codex_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")),
    )
    runner = CodexExecRunner(run_store=store, schema_path=schema)

    with pytest.raises(OSError, match="spawn failed"):
        runner.run("compose-1", "prompt")

    calls = store.rebuild_state()["calls"]
    assert calls["preflight_failure_count"] == 1
    assert calls["confirmed_external_call_count"] == 0


def test_codex_runner_recovers_completed_attempt_without_external_call(
    tmp_path, monkeypatch
) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})
    runner = CodexExecRunner(run_store=store, schema_path=schema)
    request = runner.request_for("compose-1", "prompt", {})
    attempt = store.reserve_attempt("compose-1", request, "prompt")
    (attempt / "stdout.jsonl").write_text(
        '{"type":"thread.started"}\n{"type":"turn.completed"}\n', encoding="utf-8"
    )
    (attempt / "stderr.log").write_text("", encoding="utf-8")
    (attempt / "response.staged.json").write_text(
        json.dumps({"composition_source": VALID_SOURCE, "intent_summary": "test"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._run_codex_process",
        lambda *args, **kwargs: pytest.fail("external call must not run"),
    )

    response = runner.run("compose-1", "prompt")

    assert response["composition_source"] == VALID_SOURCE
    assert json.loads((attempt / "terminal.json").read_text(encoding="utf-8"))["status"] == (
        "completed"
    )


def test_codex_runner_finalizes_invalid_recovered_response_as_failed(tmp_path, monkeypatch) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})
    runner = CodexExecRunner(run_store=store, schema_path=schema)
    request = runner.request_for("compose-1", "prompt", {})
    attempt = store.reserve_attempt("compose-1", request, "prompt")
    (attempt / "stdout.jsonl").write_text(
        '{"type":"thread.started"}\n{"type":"turn.completed"}\n', encoding="utf-8"
    )
    (attempt / "response.staged.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._run_codex_process",
        lambda *args, **kwargs: pytest.fail("external call must not run"),
    )

    with pytest.raises(ValueError, match="required object properties"):
        runner.run("compose-1", "prompt")

    terminal = json.loads((attempt / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert store.rebuild_state()["status"] == "failed"


def test_codex_runner_finalizes_response_promotion_conflict_as_failed(
    tmp_path, monkeypatch
) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})
    runner = CodexExecRunner(run_store=store, schema_path=schema)
    request = runner.request_for("compose-1", "prompt", {})
    attempt = store.reserve_attempt("compose-1", request, "prompt")
    (attempt / "stdout.jsonl").write_text(
        '{"type":"thread.started"}\n{"type":"turn.completed"}\n', encoding="utf-8"
    )
    (attempt / "response.staged.json").write_text(
        json.dumps({"composition_source": VALID_SOURCE, "intent_summary": "new"}),
        encoding="utf-8",
    )
    response_path = store.run_dir / "responses" / "compose-1.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text('{"conflict":true}', encoding="utf-8")
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._run_codex_process",
        lambda *args, **kwargs: pytest.fail("external call must not run"),
    )

    with pytest.raises(StateConflictError, match="conflicts"):
        runner.run("compose-1", "prompt")

    terminal = json.loads((attempt / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"


def test_codex_runner_recovers_response_promoted_before_terminal_write(
    tmp_path, monkeypatch
) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})

    def fake_run(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_text(
            json.dumps({"composition_source": VALID_SOURCE, "intent_summary": "test"}),
            encoding="utf-8",
        )
        kwargs["stdout"].write('{"type":"thread.started"}\n{"type":"turn.completed"}\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._resolve_codex_executable",
        lambda: (Path("codex.exe"), None),
    )
    monkeypatch.setattr("llm_musical_composer.pilot_loop._run_codex_process", fake_run)
    runner = CodexExecRunner(run_store=store, schema_path=schema)
    finalize_attempt = store.finalize_attempt
    monkeypatch.setattr(
        store,
        "finalize_attempt",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("terminal write failed")),
    )

    with pytest.raises(OSError, match="terminal write failed"):
        runner.run("compose-1", "prompt")

    assert (store.run_dir / "responses" / "compose-1.json").is_file()
    assert store.rebuild_state()["status"] == "recovery_required"
    monkeypatch.setattr(store, "finalize_attempt", finalize_attempt)
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._run_codex_process",
        lambda *args, **kwargs: pytest.fail("external call must not run during recovery"),
    )

    response = runner.run("compose-1", "prompt")

    assert response["intent_summary"] == "test"
    attempt = store.attempt_dirs("compose-1")[0]
    terminal = json.loads((attempt / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "completed"


def test_codex_runner_stops_on_ambiguous_attempt(tmp_path, monkeypatch) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})
    runner = CodexExecRunner(run_store=store, schema_path=schema)
    attempt = store.reserve_attempt(
        "compose-1", runner.request_for("compose-1", "prompt", {}), "prompt"
    )
    (attempt / "stdout.jsonl").write_text('{"type":"thread.started"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._run_codex_process",
        lambda *args, **kwargs: pytest.fail("ambiguous attempt must not be retried"),
    )

    with pytest.raises(InterruptedAttemptError):
        runner.run("compose-1", "prompt")

    assert json.loads((attempt / "terminal.json").read_text(encoding="utf-8"))["status"] == (
        "interrupted"
    )


def test_codex_runner_rejects_changed_request_and_failed_saved_attempt(tmp_path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})
    runner = CodexExecRunner(run_store=store, schema_path=schema)
    attempt = store.reserve_attempt(
        "compose-1", runner.request_for("compose-1", "prompt", {}), "prompt"
    )

    with pytest.raises(StateConflictError, match="conflicts"):
        runner.run("compose-1", "changed")

    store.finalize_attempt(attempt, "failed", returncode=1)
    with pytest.raises(RuntimeError, match="saved Codex attempt failed"):
        runner.run("compose-1", "prompt")


def test_codex_runner_recovers_turn_failed_as_terminal_failure(tmp_path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})
    runner = CodexExecRunner(run_store=store, schema_path=schema)
    attempt = store.reserve_attempt(
        "compose-1", runner.request_for("compose-1", "prompt", {}), "prompt"
    )
    (attempt / "stdout.jsonl").write_text('{"type":"turn.failed"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="saved Codex attempt failed"):
        runner.run("compose-1", "prompt")

    assert json.loads((attempt / "terminal.json").read_text(encoding="utf-8"))["status"] == (
        "failed"
    )


@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        "[]",
        '{"composition_source":"","intent_summary":"x"}',
        '{"composition_source":"x","intent_summary":301}',
    ],
)
def test_codex_runner_rejects_invalid_structured_response(tmp_path, response: str) -> None:
    path = tmp_path / "response.json"
    path.write_text(response, encoding="utf-8")

    with pytest.raises(ValueError):
        CodexExecRunner._validated_response(path)


def test_codex_runner_finalizes_missing_completed_response_as_failed(tmp_path, monkeypatch) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})

    def fake_run(command, **kwargs):
        kwargs["stdout"].write('{"type":"thread.started"}\n{"type":"turn.completed"}\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._resolve_codex_executable",
        lambda: (Path("codex.exe"), None),
    )
    monkeypatch.setattr("llm_musical_composer.pilot_loop._run_codex_process", fake_run)
    runner = CodexExecRunner(run_store=store, schema_path=schema)

    with pytest.raises(OSError):
        runner.run("compose-1", "prompt")

    attempt = store.attempt_dirs("compose-1")[0]
    terminal = json.loads((attempt / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert "response" in terminal["detail"]
    assert store.rebuild_state()["status"] == "failed"


def test_codex_runner_finalizes_invalid_completed_response_as_failed(tmp_path, monkeypatch) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})

    def fake_run(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_text("{}", encoding="utf-8")
        kwargs["stdout"].write('{"type":"thread.started"}\n{"type":"turn.completed"}\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._resolve_codex_executable",
        lambda: (Path("codex.exe"), None),
    )
    monkeypatch.setattr("llm_musical_composer.pilot_loop._run_codex_process", fake_run)
    runner = CodexExecRunner(run_store=store, schema_path=schema)

    with pytest.raises(ValueError, match="required object properties"):
        runner.run("compose-1", "prompt")

    attempt = store.attempt_dirs("compose-1")[0]
    terminal = json.loads((attempt / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert not (tmp_path / "run" / "responses" / "compose-1.json").exists()


def test_codex_runner_rejects_success_exit_without_completion_event(tmp_path, monkeypatch) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})

    def fake_run(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_text(
            json.dumps({"composition_source": VALID_SOURCE, "intent_summary": "test"}),
            encoding="utf-8",
        )
        kwargs["stdout"].write('{"type":"thread.started"}\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._resolve_codex_executable",
        lambda: (Path("codex.exe"), None),
    )
    monkeypatch.setattr("llm_musical_composer.pilot_loop._run_codex_process", fake_run)
    runner = CodexExecRunner(run_store=store, schema_path=schema)

    with pytest.raises(RuntimeError, match="no valid completion"):
        runner.run("compose-1", "prompt")


def test_codex_runner_rejects_malformed_jsonl_even_with_completion(tmp_path, monkeypatch) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})

    def fake_run(command, **kwargs):
        Path(command[command.index("-o") + 1]).write_text(
            json.dumps({"composition_source": VALID_SOURCE, "intent_summary": "test"}),
            encoding="utf-8",
        )
        kwargs["stdout"].write('{"type":"thread.started"}\nmalformed\n{"type":"turn.completed"}\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop._resolve_codex_executable",
        lambda: (Path("codex.exe"), None),
    )
    monkeypatch.setattr("llm_musical_composer.pilot_loop._run_codex_process", fake_run)
    runner = CodexExecRunner(run_store=store, schema_path=schema)

    with pytest.raises(RuntimeError, match="no valid completion"):
        runner.run("compose-1", "prompt")


def test_response_requires_composition_source() -> None:
    with pytest.raises(ValueError, match="composition_source"):
        _response_source({"intent_summary": "missing"})


@pytest.mark.parametrize(
    ("status", "expected"),
    [(RunStatus.COMPLETED, 0), (RunStatus.NO_VALID_CANDIDATE, 2)],
)
def test_main_writes_profile_and_returns_status(tmp_path, monkeypatch, status, expected) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.build_reference_profile_from_directory",
        lambda *args, **kwargs: {"axes": {}, "source_count": 1},
    )
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.build_activity_reference_profile_from_directory",
        lambda *args, **kwargs: {
            "axes": {"activity": {}},
            "source_count": 1,
            "unable_to_investigate": [],
        },
    )
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop."
        "build_material_development_reference_profile_from_directory",
        lambda *args, **kwargs: {
            **_development_profile(),
            "source_count": 1,
            "unable_to_investigate": [],
        },
    )
    runner_arguments = {}

    def fake_runner(**kwargs):
        runner_arguments.update(kwargs)
        return object()

    isolated = tmp_path / "isolated"
    monkeypatch.setattr("llm_musical_composer.pilot_loop.CodexExecRunner", fake_runner)
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.isolated_codex_working_directory",
        lambda: isolated,
    )
    monkeypatch.setattr(
        "llm_musical_composer.pilot_loop.run_pilot",
        lambda *args, **kwargs: SimpleNamespace(status=status),
    )

    result = main(["--output-root", str(tmp_path / "out")])

    assert result == expected
    run_dirs = list((tmp_path / "out").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "inputs" / "reference-profile.json").is_file()
    assert (run_dirs[0] / "inputs" / "material-development-reference-profile.json").is_file()
    assert not (tmp_path / "out" / "reference-profile.json").exists()
    assert runner_arguments["working_directory"] == isolated
