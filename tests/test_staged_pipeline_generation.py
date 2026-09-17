import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_generic_pipeline_quality import _fixture

from llm_musical_composer.generation_intent import (
    creative_targets_sha256,
    merge_generation_intent,
)
from llm_musical_composer.performance_pipeline import PiecePlan, PlanNode, ScoreSpec
from llm_musical_composer.pipeline_dsl import (
    dump_performance_spec,
    dump_piece_plan,
    dump_score_spec,
    parse_score_spec,
)
from llm_musical_composer.run_state import StateConflictError, sha256_text
from llm_musical_composer.smf_notes import load_smf_notes
from llm_musical_composer.staged_pipeline_generation import (
    _stage_target,
    run_staged_pipeline_generation,
)


class FakeRunner:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    def run(self, step_id: str, prompt: str, input_hashes=None) -> dict[str, object]:
        self.calls.append((step_id, prompt, input_hashes or {}))
        return {"composition_source": self.responses[step_id], "intent_summary": "fixture"}


def _inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    normalized = {
        "schema_version": 1,
        "preset": "solo_piano_3m_v1",
        "reference": {"mode": "unresolved_default"},
        "controls": {},
    }
    resolved = {
        "schema_version": 1,
        "preset": "solo_piano_3m_v1",
        "reference": {
            "state": "resolved",
            "name": "SecretReference.mid",
            "sha256": "a" * 64,
            "selection_method": "automatic_medoid",
        },
        "controls": {},
    }
    target = {
        "schema_version": 1,
        "target_version": "reference-generation-target-v1",
        "controls": {"brightness": 0.0, "height": 0.0, "attack_frequency": 0.0},
        "piece_plan": {
            "reference_specific_targets": [],
            "long_term_structure": {"status": "unverified"},
        },
        "stage_targets": {
            "score_spec": [{"id": "score-marker"}],
            "performance_spec": [{"id": "performance-marker"}],
            "rendered_surface": [{"id": "surface-marker"}],
        },
    }
    return normalized, resolved, target


def _responses() -> dict[str, str]:
    plan, score, performance = _fixture()
    return {
        "piece-plan": dump_piece_plan(plan),
        "score-spec-001": dump_score_spec(score),
        "performance-spec": dump_performance_spec(performance),
    }


def test_piece_plan_stage_receives_only_specified_tonal_hierarchy() -> None:
    _, _, target = _inputs()
    specified = {
        "id": "tonal_hierarchy",
        "status": "specified",
        "scale_policy": "dorian",
    }
    target["semantic_targets"] = {
        "piece_plan": [specified],
        "score_spec": [{"id": "other"}],
    }

    assert _stage_target(target, "piece_plan")["semantic_targets"] == [specified]

    target["semantic_targets"]["piece_plan"][0] = {
        "id": "tonal_hierarchy",
        "status": "unverified_continuous_reference",
    }
    assert _stage_target(target, "piece_plan")["semantic_targets"] == []


def test_piece_plan_prompt_and_step_hash_receive_creative_target(tmp_path: Path) -> None:
    normalized, resolved, target = _inputs()
    target["semantic_targets"] = {
        "piece_plan": [],
        "score_spec": [],
        "performance_spec": [],
        "rendered_surface": [],
    }
    target = merge_generation_intent(
        target,
        {
            "schema_version": 1,
            "decisions": [
                {
                    "decision_id": "climax",
                    "statement": "後半に山場を作る",
                    "decision_kind": "long_form_climax",
                    "stage_targets": [
                        {
                            "generation_stage": stage,
                            "instructions": [f"{stage}で山場を作る"],
                            "observable": {
                                "observable_id": f"climax_{stage}",
                                "scope": "whole_piece",
                                "operator": "shape",
                            },
                            "verification": "listening_only",
                        }
                        for stage in (
                            "piece_plan",
                            "harmonic_skeleton",
                            "melody_collection",
                            "texture_collection",
                            "performance_spec",
                        )
                    ],
                }
            ],
        },
    )
    runner = FakeRunner(_responses())

    run_staged_pipeline_generation(
        run_dir=tmp_path / "run",
        normalized_request=normalized,
        resolved_request=resolved,
        prompt_target=target,
        model_config={"model": "fixture-model", "reasoning_effort": "none"},
        runner=runner,
        stop_after="piece_plan",
    )

    _, prompt, hashes = runner.calls[0]
    assert "creative_climax_piece_plan" in prompt
    assert hashes["creative_targets"] == creative_targets_sha256(target, "piece_plan")


def test_piece_plan_tonal_mismatch_stops_before_score_call(tmp_path: Path) -> None:
    normalized, resolved, target = _inputs()
    plan, _, _ = _fixture()
    opposite = "major" if plan.mode == "minor" else "minor"
    target["semantic_targets"] = {
        "piece_plan": [
            {
                "id": "tonal_hierarchy",
                "status": "specified",
                "plan_mode": opposite,
            }
        ]
    }
    runner = FakeRunner(_responses())

    with pytest.raises(ValueError, match="tonal-hierarchy:plan-mode"):
        run_staged_pipeline_generation(
            run_dir=tmp_path / "run",
            normalized_request=normalized,
            resolved_request=resolved,
            prompt_target=target,
            model_config={"model": "fixture-model", "reasoning_effort": "none"},
            runner=runner,
            stop_after="score_spec",
        )

    assert [call[0] for call in runner.calls] == ["piece-plan"]


def _run(run_dir: Path, runner: FakeRunner, stop_after: str):
    normalized, resolved, target = _inputs()
    return run_staged_pipeline_generation(
        run_dir=run_dir,
        normalized_request=normalized,
        resolved_request=resolved,
        prompt_target=target,
        model_config={"model": "fixture-model", "reasoning_effort": "none"},
        runner=runner,
        stop_after=stop_after,
    )


def _assert_stage_failure(run_dir: Path, step_id: str, source: str) -> dict[str, object]:
    failure_path = run_dir / "failures" / f"{step_id}.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["schema_version"] == 1
    assert failure["step_id"] == step_id
    assert failure["composition_source"] == source
    assert failure["source_sha256"] == sha256_text(source)
    step = json.loads((run_dir / "steps" / f"{step_id}.json").read_text())
    assert step["status"] == "failed"
    assert step["outputs"]["failure_path"] == f"failures\\{step_id}.json"
    assert json.loads((run_dir / "run-state.json").read_text())["status"] == "failed"
    return failure


def test_each_stage_can_stop_and_resume_without_repeating_completed_calls(tmp_path: Path) -> None:
    first = FakeRunner(_responses())
    result = _run(tmp_path / "run", first, "piece_plan")

    assert result.status == "stopped"
    assert result.stopped_after == "piece_plan"
    assert [call[0] for call in first.calls] == ["piece-plan"]

    second = FakeRunner(_responses())
    result = _run(tmp_path / "run", second, "score_spec")

    assert result.status == "stopped"
    assert result.stopped_after == "score_spec"
    assert [call[0] for call in second.calls] == ["score-spec-001"]

    third = FakeRunner(_responses())
    result = _run(tmp_path / "run", third, "performance_spec")

    assert result.status == "completed"
    assert [call[0] for call in third.calls] == ["performance-spec"]
    assert result.musicxml_path is not None and result.musicxml_path.is_file()
    assert result.smf_path is not None and result.smf_path.read_bytes()[:4] == b"MThd"
    assert result.quality["passes"] is True
    assert len(load_smf_notes(result.smf_path)) == 18


def test_valid_noncanonical_model_formatting_is_saved_as_canonical_dsl(tmp_path: Path) -> None:
    responses = _responses()
    canonical = responses["piece-plan"]
    responses["piece-plan"] = canonical.replace(", ", ",\n", 1)

    result = _run(tmp_path / "run", FakeRunner(responses), "piece_plan")

    assert result.piece_plan_path.read_text(encoding="utf-8") == canonical + "\n"


def test_stage_prompts_receive_only_their_anonymous_target_views(tmp_path: Path) -> None:
    runner = FakeRunner(_responses())

    _run(tmp_path / "run", runner, "performance_spec")

    prompts = {step: prompt for step, prompt, _ in runner.calls}
    assert "`tonal_center`は0から11の整数" in prompts["piece-plan"]
    assert "`harmonic_focus`は0から11の整数またはNone" in prompts["piece-plan"]
    assert "duration_weightは1以上の整数" in prompts["piece-plan"]
    assert "0.09のような小数や比率" in prompts["piece-plan"]
    assert "score-marker" not in prompts["piece-plan"]
    assert "performance-marker" not in prompts["piece-plan"]
    assert "score-marker" in prompts["score-spec-001"]
    assert "surface-marker" in prompts["score-spec-001"]
    assert "duration_units * 2 <= divisions" in prompts["score-spec-001"]
    assert (
        "note.at_units <= onset < note.at_units + note.duration_units"
        in prompts["score-spec-001"]
    )
    assert "同じ実音高を重複なしの集合" in prompts["score-spec-001"]
    assert "2番目の音高から最低音を引いた値" in prompts["score-spec-001"]
    assert "和音構成音どうしでも実音高差が5半音なら不合格" in prompts[
        "score-spec-001"
    ]
    assert "実音高差が7半音以上なら禁止しません" in prompts["score-spec-001"]
    assert "major={0,4,7}" in prompts["score-spec-001"]
    assert "minor={0,3,7}" in prompts["score-spec-001"]
    assert "diminished={0,3,6}" in prompts["score-spec-001"]
    assert "major-seventh={0,4,7,11}" in prompts["score-spec-001"]
    assert "root_pitch_classを加えて12で割った余り" in prompts["score-spec-001"]
    assert (
        "harmony.at_units < note.at_units + note.duration_units"
        in prompts["score-spec-001"]
    )
    assert (
        "note.at_units < harmony.at_units + harmony.duration_units"
        in prompts["score-spec-001"]
    )
    assert "note.pitch % 12" in prompts["score-spec-001"]
    assert "重なるすべての和声" in prompts["score-spec-001"]
    assert "音価を和声境界までに短くする" in prompts["score-spec-001"]
    assert "境界後を別のscore_note" in prompts["score-spec-001"]
    assert "発音時点の和声だけ" in prompts["score-spec-001"]
    for v7_specific_text in (
        "material_5",
        "material_6",
        "MIDI 24",
        "B minor",
        "m5_l07",
        "m5_l10",
        "m6_l4",
    ):
        assert v7_specific_text not in prompts["score-spec-001"]
    assert "単一和声の専用ScoreMaterial" in prompts["score-spec-001"]
    assert "performance-marker" in prompts["performance-spec"]
    assert "surface-marker" in prompts["performance-spec"]
    assert "派生元の葉と同じscore_material_id" in prompts["piece-plan"]
    assert "最終葉にはderived_fromを設定しない" in prompts["piece-plan"]
    assert "単一和声素材どうしの境界" in prompts["performance-spec"]
    assert "最も近い明示指定" in prompts["performance-spec"]
    assert "一つのペダル区間へ統合" in prompts["performance-spec"]
    assert "和声を持つ葉ではharmony_legatoを基本" in prompts["performance-spec"]
    assert '"selectable_values"' in prompts["performance-spec"]
    assert '"clear"' not in prompts["performance-spec"]
    assert "実効coordination_profileはscoreまたはaligned" in prompts["performance-spec"]
    assert '"node_id": "tonic-ending"' in prompts["performance-spec"]
    assert "event_id='e-u0'" not in prompts["performance-spec"]
    assert all("SecretReference.mid" not in prompt for prompt in prompts.values())
    assert all("a" * 64 not in prompt for prompt in prompts.values())


def test_saved_output_or_model_change_stops_instead_of_regenerating(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _run(run_dir, FakeRunner(_responses()), "piece_plan")
    (run_dir / "outputs" / "piece-plan.dsl").write_text("changed", encoding="utf-8")

    with pytest.raises(StateConflictError, match="output hash"):
        _run(run_dir, FakeRunner(_responses()), "piece_plan")

    other = tmp_path / "other"
    normalized, resolved, target = _inputs()
    _run(other, FakeRunner(_responses()), "piece_plan")
    with pytest.raises(StateConflictError, match="run-spec"):
        run_staged_pipeline_generation(
            run_dir=other,
            normalized_request=normalized,
            resolved_request=resolved,
            prompt_target=target,
            model_config={"model": "different-model"},
            runner=FakeRunner(_responses()),
            stop_after="piece_plan",
        )


def test_invalid_piece_plan_stops_before_score_call(tmp_path: Path) -> None:
    responses = _responses()
    responses["piece-plan"] = responses["piece-plan"].replace(
        "derived_from='first-scene'", "derived_from='missing'"
    )
    runner = FakeRunner(responses)

    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="earlier plan node"):
        _run(run_dir, runner, "performance_spec")

    assert [call[0] for call in runner.calls] == ["piece-plan"]
    _assert_stage_failure(run_dir, "piece-plan", responses["piece-plan"])
    assert not (run_dir / "outputs" / "piece-plan.dsl").exists()

    retry = FakeRunner(_responses())
    with pytest.raises(StateConflictError, match="failed"):
        _run(run_dir, retry, "piece_plan")
    assert retry.calls == []


def test_piece_plan_dsl_failure_is_saved_and_not_retried(tmp_path: Path) -> None:
    responses = _responses()
    responses["piece-plan"] = responses["piece-plan"].replace(
        "duration_weight=1", "duration_weight=1.0", 1
    )
    runner = FakeRunner(responses)
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="expected a literal int"):
        _run(run_dir, runner, "piece_plan")

    failure = _assert_stage_failure(run_dir, "piece-plan", responses["piece-plan"])
    assert failure["error_type"] == "PipelineDslError"
    step = json.loads((run_dir / "steps" / "piece-plan.json").read_text())
    assert step["input_hashes"] == runner.calls[0][2]
    assert not (run_dir / "outputs" / "piece-plan.dsl").exists()

    retry = FakeRunner(_responses())
    with pytest.raises(StateConflictError, match="failed"):
        _run(run_dir, retry, "piece_plan")
    assert retry.calls == []


def test_piece_plan_without_contrast_is_not_saved_or_sent_to_score_stage(
    tmp_path: Path,
) -> None:
    plan, _, _ = _fixture()
    responses = _responses()
    responses["piece-plan"] = dump_piece_plan(
        replace(
            plan,
            nodes=tuple(replace(node, contrasts_with=None) for node in plan.nodes),
        )
    )
    runner = FakeRunner(responses)

    with pytest.raises(ValueError, match=r"piece plan quality failed.*missing-contrast"):
        _run(tmp_path / "run", runner, "performance_spec")

    assert [call[0] for call in runner.calls] == ["piece-plan"]
    assert not (tmp_path / "run" / "outputs" / "piece-plan.dsl").exists()
    quality = json.loads(
        (tmp_path / "run" / "outputs" / "piece-plan-quality.json").read_text()
    )
    assert "missing-contrast" in quality["failures"]
    quality_step = json.loads(
        (tmp_path / "run" / "steps" / "piece-plan-quality.json").read_text()
    )
    assert quality_step["status"] == "failed"
    assert json.loads((tmp_path / "run" / "run-state.json").read_text())["status"] == "failed"


def test_unassessable_piece_plan_recurrence_is_failed_before_score_call(
    tmp_path: Path,
) -> None:
    plan, _, _ = _fixture()
    responses = _responses()
    responses["piece-plan"] = dump_piece_plan(
        replace(
            plan,
            nodes=tuple(
                replace(node, score_material_id="contrast")
                if node.node_id == "recollected-scene"
                else node
                for node in plan.nodes
            ),
        )
    )
    runner = FakeRunner(responses)
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="unassessable-recurrence:recollected-scene"):
        _run(run_dir, runner, "performance_spec")

    assert [call[0] for call in runner.calls] == ["piece-plan"]
    assert not (run_dir / "outputs" / "piece-plan.dsl").exists()
    assert json.loads((run_dir / "run-state.json").read_text())["status"] == "failed"

    retry = FakeRunner(_responses())
    with pytest.raises(StateConflictError, match="failed"):
        _run(run_dir, retry, "performance_spec")
    assert retry.calls == []


def test_invalid_score_spec_stops_before_performance_call(tmp_path: Path) -> None:
    responses = _responses()
    responses["score-spec-001"] = responses["score-spec-001"].replace(
        "material_id='theme'", "material_id='unexpected'", 1
    )
    runner = FakeRunner(responses)

    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="returned material ids"):
        _run(run_dir, runner, "performance_spec")

    assert [call[0] for call in runner.calls] == ["piece-plan", "score-spec-001"]
    _assert_stage_failure(run_dir, "score-spec-001", responses["score-spec-001"])
    assert not (run_dir / "outputs" / "score-spec-001.dsl").exists()

    retry = FakeRunner(_responses())
    with pytest.raises(StateConflictError, match="failed"):
        _run(run_dir, retry, "score_spec")
    assert retry.calls == []


@pytest.mark.parametrize(
    ("step_id", "replacement", "error_match"),
    [
        ("score-spec-001", ("score_spec(", "unknown_score("), "unknown DSL function"),
        (
            "performance-spec",
            ("performance_spec(", "unknown_performance("),
            "unknown DSL function",
        ),
        (
            "performance-spec",
            ("timing_budget_id='narrative-v2'", "timing_budget_id='invalid'"),
            "performance metadata",
        ),
    ],
)
def test_stage_dsl_or_semantic_failure_is_saved(
    tmp_path: Path,
    step_id: str,
    replacement: tuple[str, str],
    error_match: str,
) -> None:
    responses = _responses()
    responses[step_id] = responses[step_id].replace(*replacement, 1)
    runner = FakeRunner(responses)
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match=error_match):
        _run(run_dir, runner, "performance_spec")

    _assert_stage_failure(run_dir, step_id, responses[step_id])
    assert not (run_dir / "outputs" / f"{step_id}.dsl").exists()
    assert not (run_dir / "outputs" / "final.mid").exists()


def test_score_semantic_failure_is_rejected_before_batch_is_saved(tmp_path: Path) -> None:
    responses = _responses()
    responses["score-spec-001"] = responses["score-spec-001"].replace(
        "pitch=69", "pitch=109", 1
    )
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="supported piano range"):
        _run(run_dir, FakeRunner(responses), "score_spec")

    _assert_stage_failure(run_dir, "score-spec-001", responses["score-spec-001"])
    assert not (run_dir / "outputs" / "score-spec-001.dsl").exists()
    assert not (run_dir / "outputs" / "score-spec.dsl").exists()


@pytest.mark.parametrize("mutation", ["missing_artifact", "missing_step", "changed_content"])
def test_inconsistent_stage_failure_stops_before_runner(
    tmp_path: Path, mutation: str
) -> None:
    responses = _responses()
    responses["piece-plan"] = responses["piece-plan"].replace(
        "duration_weight=1", "duration_weight=1.0", 1
    )
    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="expected a literal int"):
        _run(run_dir, FakeRunner(responses), "piece_plan")

    failure_path = run_dir / "failures" / "piece-plan.json"
    step_path = run_dir / "steps" / "piece-plan.json"
    if mutation == "missing_artifact":
        failure_path.unlink()
    elif mutation == "missing_step":
        step_path.unlink()
    else:
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        failure["composition_source"] += " "
        failure_path.write_text(json.dumps(failure), encoding="utf-8")

    retry = FakeRunner(_responses())
    with pytest.raises(StateConflictError, match="stage failure"):
        _run(run_dir, retry, "piece_plan")
    assert retry.calls == []


def test_unexpected_implementation_error_is_not_saved_as_model_failure(
    tmp_path: Path, monkeypatch
) -> None:
    def fail_unexpectedly(source: str):
        del source
        raise RuntimeError("implementation defect")

    monkeypatch.setattr(
        "llm_musical_composer.staged_pipeline_generation.parse_piece_plan",
        fail_unexpectedly,
    )
    run_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="implementation defect"):
        _run(run_dir, FakeRunner(_responses()), "piece_plan")

    assert not (run_dir / "failures" / "piece-plan.json").exists()
    assert not (run_dir / "steps" / "piece-plan.json").exists()
    assert json.loads((run_dir / "run-state.json").read_text())["status"] == "initialized"


def test_score_quality_failure_is_recorded_before_performance_call(tmp_path: Path) -> None:
    responses = _responses()
    responses["score-spec-001"] = responses["score-spec-001"].replace(
        "event_id='t-u0', at_units=0, duration_units=4, pitch=69",
        "event_id='t-u0', at_units=0, duration_units=4, pitch=74",
    )
    runner = FakeRunner(responses)
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="score quality failed"):
        _run(run_dir, runner, "performance_spec")

    assert [call[0] for call in runner.calls] == ["piece-plan", "score-spec-001"]
    assert not (run_dir / "outputs" / "score-spec.dsl").exists()
    quality_path = run_dir / "outputs" / "score-quality.json"
    assert quality_path.is_file()
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert "unsupported-dissonance:theme" in quality["failures"]
    step = json.loads((run_dir / "steps" / "score-quality.json").read_text())
    assert step["status"] == "failed"
    state = json.loads((run_dir / "run-state.json").read_text())
    assert state["status"] == "failed"

    retry = FakeRunner(_responses())
    with pytest.raises(StateConflictError, match="failed"):
        _run(run_dir, retry, "performance_spec")
    assert retry.calls == []


def test_quality_failure_does_not_publish_final_artifacts(tmp_path: Path) -> None:
    responses = _responses()
    responses["performance-spec"] = responses["performance-spec"].replace(
        "target_duration_ms=180000", "target_duration_ms=179000"
    )
    runner = FakeRunner(responses)

    with pytest.raises(ValueError, match="generic quality gate failed"):
        _run(tmp_path / "run", runner, "performance_spec")

    assert [call[0] for call in runner.calls] == [
        "piece-plan",
        "score-spec-001",
        "performance-spec",
    ]
    assert not (tmp_path / "run" / "steps" / "publish-final.json").exists()
    assert not (tmp_path / "run" / "outputs" / "performance-spec.dsl").exists()
    quality_step = json.loads((tmp_path / "run" / "steps" / "performance-quality.json").read_text())
    assert quality_step["status"] == "failed"
    state = json.loads((tmp_path / "run" / "run-state.json").read_text())
    assert state["status"] == "failed"
    assert not (tmp_path / "run" / "outputs" / "final.mid").exists()


def test_render_error_is_recorded_as_failed_performance_quality(tmp_path: Path) -> None:
    plan, score, _ = _fixture()
    theme, contrast, ending = score.materials
    responses = _responses()
    responses["score-spec-001"] = dump_score_spec(
        replace(
            score,
            materials=(
                replace(theme, harmonies=(), foreground_voice=None),
                contrast,
                ending,
            ),
        )
    )
    runner = FakeRunner(responses)
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="performance quality failed"):
        _run(run_dir, runner, "performance_spec")

    quality = json.loads((run_dir / "outputs" / "performance-quality.json").read_text())
    assert quality["passes"] is False
    assert quality["error_type"] == "PipelineValidationError"
    assert not (run_dir / "outputs" / "performance-spec.dsl").exists()
    assert json.loads((run_dir / "run-state.json").read_text())["status"] == "failed"
    assert plan.plan_id == "generic-plan"


def test_final_rolled_coordination_is_rejected_before_canonical_performance(
    tmp_path: Path,
) -> None:
    responses = _responses()
    responses["performance-spec"] = responses["performance-spec"].replace(
        "node_id='tonic-ending', timing_profile='release', timing_amount='subtle', "
        "dynamics_profile='release', articulation_profile='legato', "
        "coordination_profile='aligned'",
        "node_id='tonic-ending', timing_profile='release', timing_amount='subtle', "
        "dynamics_profile='release', articulation_profile='legato', "
        "coordination_profile='rolled'",
    )
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="generic quality gate failed"):
        _run(run_dir, FakeRunner(responses), "performance_spec")

    quality = json.loads((run_dir / "outputs" / "performance-quality.json").read_text())
    assert "final-tonic-coordination" in quality["failures"]
    assert not (run_dir / "outputs" / "performance-spec.dsl").exists()


def test_orphan_or_changed_quality_output_stops_before_runner(tmp_path: Path) -> None:
    orphan_dir = tmp_path / "orphan"
    _run(orphan_dir, FakeRunner(_responses()), "piece_plan")
    quality_path = orphan_dir / "outputs" / "score-quality.json"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text("{}", encoding="utf-8")
    orphan_runner = FakeRunner(_responses())

    with pytest.raises(StateConflictError, match="must coexist"):
        _run(orphan_dir, orphan_runner, "score_spec")
    assert orphan_runner.calls == []

    changed_dir = tmp_path / "changed"
    _run(changed_dir, FakeRunner(_responses()), "score_spec")
    score_quality_path = changed_dir / "outputs" / "score-quality.json"
    score_quality_path.write_text("{}", encoding="utf-8")
    changed_runner = FakeRunner(_responses())

    with pytest.raises(StateConflictError, match="quality hash"):
        _run(changed_dir, changed_runner, "score_spec")
    assert changed_runner.calls == []


def _batched_score_responses() -> tuple[PiecePlan, ScoreSpec, ScoreSpec]:
    base_plan, base_score, _ = _fixture()
    root = base_plan.nodes[0]
    nodes = [root]
    materials = []
    for index in range(10):
        material_index = 0 if index == 7 else index - 1 if index > 7 else index
        material_id = f"material-{material_index}"
        nodes.append(
            PlanNode(
                f"scene-{index}",
                root.node_id,
                index,
                "contrast"
                if index == 1
                else "return"
                if index == 7
                else "release"
                if index == 9
                else "statement",
                derived_from="scene-0" if index == 7 else None,
                contrasts_with="scene-0" if index == 1 else None,
                duration_weight=1,
                score_material_id=material_id,
            )
        )
        if any(material.material_id == material_id for material in materials):
            continue
        original = base_score.materials[material_index % 2]
        materials.append(
            replace(
                original,
                material_id=material_id,
                notes=tuple(
                    replace(note, event_id=f"{material_id}-{note.event_id}")
                    for note in original.notes
                ),
                harmonies=tuple(
                    replace(harmony, harmony_id=f"{material_id}-{harmony.harmony_id}")
                    for harmony in original.harmonies
                ),
            )
        )
    plan = replace(base_plan, nodes=tuple(nodes))
    first = ScoreSpec(base_score.score_id, base_score.divisions, tuple(materials[:8]))
    second = ScoreSpec(base_score.score_id, base_score.divisions, tuple(materials[8:]))
    return plan, first, second


def test_more_than_eight_materials_are_generated_in_deterministic_batches(
    tmp_path: Path,
) -> None:
    plan, first, second = _batched_score_responses()
    responses = {
        "piece-plan": dump_piece_plan(plan),
        "score-spec-001": dump_score_spec(first),
        "score-spec-002": dump_score_spec(second),
    }
    runner = FakeRunner(responses)
    normalized, resolved, target = _inputs()

    result = run_staged_pipeline_generation(
        run_dir=tmp_path / "run",
        normalized_request=normalized,
        resolved_request=resolved,
        prompt_target=target,
        model_config={"model": "fixture-model"},
        runner=runner,
        stop_after="score_spec",
    )

    assert result.status == "stopped"
    assert [call[0] for call in runner.calls] == [
        "piece-plan",
        "score-spec-001",
        "score-spec-002",
    ]
    assert result.score_spec_path is not None
    saved_score = parse_score_spec(result.score_spec_path.read_text(encoding="utf-8"))
    assert [material.material_id for material in saved_score.materials] == [
        f"material-{index}" for index in range(9)
    ]


@pytest.mark.parametrize(
    ("field", "value", "error_match"),
    [
        ("score_id", "different-score", "same score_id"),
        ("divisions", 99, "same divisions"),
    ],
)
def test_second_score_batch_metadata_mismatch_is_failed(
    tmp_path: Path, field: str, value: object, error_match: str
) -> None:
    plan, first, second = _batched_score_responses()
    invalid_second = replace(second, **{field: value})
    responses = {
        "piece-plan": dump_piece_plan(plan),
        "score-spec-001": dump_score_spec(first),
        "score-spec-002": dump_score_spec(invalid_second),
    }
    runner = FakeRunner(responses)
    normalized, resolved, target = _inputs()
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match=error_match):
        run_staged_pipeline_generation(
            run_dir=run_dir,
            normalized_request=normalized,
            resolved_request=resolved,
            prompt_target=target,
            model_config={"model": "fixture-model"},
            runner=runner,
            stop_after="score_spec",
        )

    _assert_stage_failure(run_dir, "score-spec-002", responses["score-spec-002"])
    assert (run_dir / "outputs" / "score-spec-001.dsl").is_file()
    assert not (run_dir / "outputs" / "score-spec-002.dsl").exists()
    assert not (run_dir / "outputs" / "score-spec.dsl").exists()
