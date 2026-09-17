from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from llm_musical_composer.composition_ir import Material, Note, Pedal
from llm_musical_composer.long_form_generation import (
    LongFormGenerationError,
    _material_source,
    _validate_long_form_inner_structure,
    _validate_long_form_phrase_structure,
    _validate_natural_long_form_plan,
    composition_to_source,
)
from llm_musical_composer.long_form_loop import (
    PresetPlanRunner,
    PresetResponseRunner,
    build_plan_prompt,
    evaluate_long_form_candidate,
    load_plan_source,
    load_preset_responses,
    load_verified_style_target,
    main,
    run_long_form,
)
from llm_musical_composer.music_dsl import THREE_MINUTE_POLICY, parse_composition
from llm_musical_composer.run_state import sha256_file
from llm_musical_composer.smf_render import render_composition
from tests.test_music_dsl import LONG_PARTS_SOURCE


def _write_target(
    tmp_path: Path, *, density_range: tuple[float, float] = (1.0, 3.0)
) -> tuple[Path, Path, Path]:
    records = tmp_path / "files.jsonl"
    records.write_text('{"name":"a.mid"}\n', encoding="utf-8")
    target = tmp_path / "style-target.json"
    target.write_text(
        json.dumps(
            {
                "status": "pass",
                "selection": {"anchor_name": "secret.mid"},
                "prompt_target": {
                    "schema_version": 1,
                    "scope": "neighbor level ranges only",
                    "axis_targets": {
                        "density": {
                            "level": {
                                "range_low": density_range[0],
                                "median": 2.0,
                                "range_high": density_range[1],
                            }
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sha256": {
                    "style_target": sha256_file(target),
                    "input_jsonl": sha256_file(records),
                }
            }
        ),
        encoding="utf-8",
    )
    return target, manifest, records


def test_verified_target_checks_both_hashes_and_returns_prompt_only(tmp_path: Path) -> None:
    target_path, manifest_path, records_path = _write_target(tmp_path)

    verified = load_verified_style_target(target_path, manifest_path, records_path)

    assert verified["target"]["selection"]["anchor_name"] == "secret.mid"
    assert verified["prompt_target"]["axis_targets"]["density"]["level"]["median"] == 2
    assert verified["hashes"]["style_target"] == sha256_file(target_path)
    assert verified["feasibility"]["status"] == "pass"

    records_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reference records fingerprint"):
        load_verified_style_target(target_path, manifest_path, records_path)


def test_verified_target_rejects_target_tampering(tmp_path: Path) -> None:
    target_path, manifest_path, records_path = _write_target(tmp_path)
    target_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="style target fingerprint"):
        load_verified_style_target(target_path, manifest_path, records_path)


def test_verified_target_reports_unreachable_density_before_generation(tmp_path: Path) -> None:
    target_path, manifest_path, records_path = _write_target(
        tmp_path, density_range=(6.12745098, 12.54863372)
    )

    verified = load_verified_style_target(target_path, manifest_path, records_path)

    assert verified["feasibility"]["status"] == "conflict"
    assert verified["feasibility"]["checks"]["density_level"]["minimum_required_note_count"] == 1103


def test_long_form_run_stops_before_creating_state_for_an_unreachable_target(
    tmp_path: Path,
) -> None:
    target_path, manifest_path, records_path = _write_target(
        tmp_path, density_range=(6.12745098, 12.54863372)
    )

    with pytest.raises(ValueError, match="requires at least 1103 notes"):
        run_long_form(
            source_dir=tmp_path,
            target_path=target_path,
            manifest_path=manifest_path,
            reference_records_path=records_path,
            schema_path=tmp_path / "unused-schema.json",
            output_root=tmp_path / "runs",
            run_id="unreachable",
            model="unused-model",
        )

    assert not (tmp_path / "runs/unreachable").exists()


def test_verified_target_rejects_missing_manifest_hashes_and_unusable_target(
    tmp_path: Path,
) -> None:
    target_path, manifest_path, records_path = _write_target(tmp_path)
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no sha256 object"):
        load_verified_style_target(target_path, manifest_path, records_path)

    target_path.write_text(json.dumps({"status": "fail"}), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "sha256": {
                    "style_target": sha256_file(target_path),
                    "input_jsonl": sha256_file(records_path),
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not usable"):
        load_verified_style_target(target_path, manifest_path, records_path)


def test_plan_prompt_contains_only_numeric_prompt_target(tmp_path: Path) -> None:
    target_path, manifest_path, records_path = _write_target(tmp_path)
    verified = load_verified_style_target(target_path, manifest_path, records_path)

    prompt = build_plan_prompt(
        "target={{style_target}} brief={{plan_brief}}",
        verified["prompt_target"],
        plan_brief="four asymmetric parts",
    )

    assert "density" in prompt
    assert "four asymmetric parts" in prompt
    assert "secret.mid" not in prompt
    assert "selection" not in prompt


def test_plan_prompt_rejects_unresolved_variables() -> None:
    with pytest.raises(ValueError, match="unresolved"):
        build_plan_prompt("{{style_target}} {{other}}", {"value": 1})


def test_live_plan_prompt_strips_the_complete_validation_example() -> None:
    template = Path("prompts/long-form-plan.md").read_text(encoding="utf-8")

    prompt = build_plan_prompt(
        template,
        {"axis_targets": {}},
        plan_brief="use four unequal parts and a late non-adjacent return",
    )

    assert "use four unequal parts" in prompt
    assert "INTERNAL_VALIDATION_EXAMPLE" not in prompt
    assert 'material("A", duration_ms=5500, notes=[])' not in prompt
    assert 'part("P1"' not in prompt
    assert 'part("PART_ID", role=..., energy=..., phrases=[...])' in prompt
    assert 'phrase("PHRASE_ID", role=..., uses=[use(...)])' in prompt
    assert 'phrase("PHRASE_ID", role=..., form=[use(...)])' not in prompt
    assert "C=0、C#=1、D=2" in prompt
    assert "素材生成は最大4バッチ" in prompt
    assert "派生素材からさらに派生させず" in prompt


def test_saved_structured_plan_can_be_reused_without_an_external_plan_call(
    tmp_path: Path,
) -> None:
    response_path = tmp_path / "plan.json"
    response_path.write_text(
        json.dumps({"composition_source": "composition(...)"}), encoding="utf-8"
    )

    class Delegate:
        def __init__(self) -> None:
            self.calls = []

        def run(self, step_id, prompt, input_hashes=None):
            self.calls.append((step_id, prompt, input_hashes))
            return {"composition_source": "unused", "intent_summary": "unused"}

    delegate = Delegate()
    runner = PresetPlanRunner(delegate, load_plan_source(response_path))

    response = runner.run("plan", "ignored", {"input": "hash"})

    assert response["composition_source"] == "composition(...)"
    assert delegate.calls == []
    assert runner.run("material-batch-01", "live")["composition_source"] == "unused"
    assert [call[0] for call in delegate.calls] == ["material-batch-01"]


def test_saved_plan_must_have_a_string_composition_source(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text('{"composition_source": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="composition_source"):
        load_plan_source(path)

    source_path = tmp_path / "plan.music.py"
    source_path.write_text("composition(...)\n", encoding="utf-8")
    assert load_plan_source(source_path) == "composition(...)\n"


def test_saved_responses_are_reused_and_only_missing_steps_are_delegated(
    tmp_path: Path,
) -> None:
    response_dir = tmp_path / "responses"
    response_dir.mkdir()
    saved = {
        "composition_source": "material_batch(...)",
        "intent_summary": "saved",
    }
    (response_dir / "material-batch-01.json").write_text(json.dumps(saved), encoding="utf-8")
    (response_dir / "material-repair-variations.json").write_text(
        json.dumps(saved), encoding="utf-8"
    )

    class Delegate:
        def __init__(self) -> None:
            self.calls = []

        def run(self, step_id, prompt, input_hashes=None):
            self.calls.append((step_id, prompt, input_hashes))
            return {"composition_source": "live", "intent_summary": "live"}

    delegate = Delegate()
    runner = PresetResponseRunner(delegate, load_preset_responses(response_dir))

    assert runner.run("material-batch-01", "ignored") == saved
    assert runner.run("material-repair-variations", "ignored") == saved
    assert runner.run("material-batch-02", "live prompt")["composition_source"] == "live"
    assert [call[0] for call in delegate.calls] == ["material-batch-02"]


def test_saved_response_directory_rejects_invalid_or_unknown_responses(tmp_path: Path) -> None:
    response_dir = tmp_path / "responses"
    response_dir.mkdir()
    (response_dir / "unknown.json").write_text(
        json.dumps({"composition_source": "x", "intent_summary": "x"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown preset response"):
        load_preset_responses(response_dir)

    (response_dir / "unknown.json").unlink()
    (response_dir / "plan.json").write_text(
        json.dumps({"composition_source": 1, "intent_summary": "x"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="composition_source"):
        load_preset_responses(response_dir)

    (response_dir / "plan.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_preset_responses(response_dir)

    (response_dir / "plan.json").unlink()
    with pytest.raises(ValueError, match="contains no usable responses"):
        load_preset_responses(response_dir)


def test_live_plan_prompt_documents_the_exact_dsl_shape() -> None:
    template = Path("prompts/long-form-plan.md").read_text(encoding="utf-8")

    assert 'composition(\n    title="曲名",' in template
    assert '"P1"' in template
    assert "phrases=[" in template
    assert 'role="variation"' in template
    assert 'variation_kind="rhythmic"' in template
    assert 'derived_from="A1"' in template
    assert 'role="opening"' in template
    assert 'material("A", duration_ms=5500, notes=[])' in template
    assert 'use("T1", role="transition", energy=2)' in template
    assert "60% 以上" in template
    assert "schema_version、scope、axis_targets を composition の引数へ入れない" in template

    example = template.split("```python\n", 1)[1].split("\n```", 1)[0]
    composition = parse_composition(
        example,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    _validate_long_form_inner_structure(composition)
    _validate_long_form_phrase_structure(composition)
    _validate_natural_long_form_plan(composition)


def test_material_prompt_documents_safe_explicit_event_syntax() -> None:
    template = Path("prompts/long-form-material-batch.md").read_text(encoding="utf-8")

    assert 'note("A-n001", 60, 100, 0, 400)' in template
    assert 'voice="upper"' in template
    assert 'cc64("A-p001", at_ms=0, value=127)' in template
    assert "リスト内包表記" in template
    assert "変数、計算式" in template
    assert "allowed_pitch_classes" in template
    assert "pitch 62 と 64" in template
    assert "pitch 61、64、67" in template
    assert "導音を使う場合は同時刻に他の音を置かず" in template
    assert "同じ pitch と at_ms の note は一件だけ" in template
    assert "transition_to" in template
    assert "source_material" in template
    assert "同じ `derived_from`" in template
    assert "upper の発音位置の少なくとも 3 分の 1" in template
    assert "lower の音程または発音位置の少なくとも 3 分の 1" in template
    assert "duration_ms と同じ時刻" in template
    assert "単純な音数や音量の別名ではありません" in template
    assert "これらは最大化する値ではありません" in template
    assert "少なくとも二つを明確に高く" not in template


def _material(material_id: str, count: int, velocity: int, chord_size: int) -> Material:
    return Material(
        material_id,
        44_000,
        tuple(
            Note(
                event_id=f"{material_id.lower()}-note-{index}",
                at_ms=(index // chord_size) * 2_000,
                duration_ms=800 + (index % 3) * 200,
                pitch=48 + index,
                velocity=velocity + (index % 2),
            )
            for index in range(count)
        ),
        (
            Pedal(f"{material_id.lower()}-pedal-1", 0, 127),
            Pedal(f"{material_id.lower()}-pedal-2", 20_000, 0),
            Pedal(f"{material_id.lower()}-pedal-3", 20_200, 127),
            Pedal(f"{material_id.lower()}-pedal-4", 43_800, 0),
        ),
    )


def _composition():
    plan = parse_composition(
        LONG_PARTS_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    return replace(
        plan,
        materials=(
            _material("A", 4, 50, 1),
            _material("B", 8, 68, 2),
            _material("C", 16, 92, 4),
        ),
    )


def _bounds() -> dict[str, object]:
    return {
        "minimum": -1_000,
        "p25": -1_000,
        "median": 0,
        "p75": 1_000,
        "maximum": 1_000,
        "target_low": -1_000,
        "target_high": 1_000,
    }


def _full_target() -> dict[str, object]:
    return {
        "axes": {
            axis: {"level": _bounds(), "shape": [_bounds() for _ in range(4)]}
            for axis in ("density", "polyphony", "velocity", "register")
        }
    }


def test_candidate_evaluation_keeps_hard_and_diagnostic_metrics_separate(
    tmp_path: Path,
) -> None:
    composition = _composition()
    midi_path = tmp_path / "candidate.mid"
    render_composition(composition, midi_path)
    profile = {
        "metrics": {
            name: {"p25": 0, "median": 0.5, "p75": 1}
            for name in (
                "duration_change_rate",
                "ioi_change_rate",
                "attack_size_change_rate",
            )
        }
    }

    report = evaluate_long_form_candidate(
        composition,
        midi_path,
        style_target=_full_target(),
        material_development_profile=profile,
    )

    assert report["status"] == "pass"
    assert all(report["hard_gates"].values())
    assert report["smf_reread"]["end_tick"] == 180_000
    assert report["smf_reread"]["cc64_count"] > 0
    assert report["style_target"]["status"] == "pass"
    assert report["metric_policy"]["diagnostic_only"] == [
        "style_target",
        "material_development",
    ]


def test_candidate_evaluation_reports_ending_and_style_investigation_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    composition = _composition()
    midi_path = tmp_path / "candidate.mid"
    render_composition(composition, midi_path)

    def fail_ending(*args, **kwargs):
        raise ValueError("ending failed")

    def fail_style(*args, **kwargs):
        raise ValueError("style failed")

    monkeypatch.setattr(
        "llm_musical_composer.long_form_loop._validate_rendered_ending", fail_ending
    )
    monkeypatch.setattr("llm_musical_composer.long_form_loop.extract_style_features", fail_style)

    report = evaluate_long_form_candidate(
        composition,
        midi_path,
        style_target=_full_target(),
        material_development_profile={
            "metrics": {
                name: {"p25": 0}
                for name in (
                    "duration_change_rate",
                    "ioi_change_rate",
                    "attack_size_change_rate",
                )
            }
        },
    )

    assert report["status"] == "fail"
    assert report["tonic_ending"]["issues"] == ["ending failed"]
    assert report["style_target"]["status"] == "unable_to_investigate"


def test_candidate_evaluation_can_make_naturalness_a_hard_gate(tmp_path: Path) -> None:
    composition = _composition()
    midi_path = tmp_path / "candidate.mid"
    render_composition(composition, midi_path)
    profile = {
        "metrics": {
            name: {"p25": 0}
            for name in (
                "duration_change_rate",
                "ioi_change_rate",
                "attack_size_change_rate",
            )
        }
    }

    report = evaluate_long_form_candidate(
        composition,
        midi_path,
        style_target=_full_target(),
        material_development_profile=profile,
        require_naturalness=True,
    )

    assert report["hard_gates"]["naturalness"] is False
    assert report["naturalness"]["status"] == "fail"
    assert report["status"] == "fail"


class _FakeCodexRunner:
    composition = _composition()

    def __init__(self, **kwargs) -> None:
        self.run_store = kwargs["run_store"]

    def run(self, step_id: str, prompt: str, input_hashes=None):
        if step_id == "plan":
            source = LONG_PARTS_SOURCE
        else:
            material_id = {
                "material-batch-01": "A",
                "material-batch-02": "B",
                "material-batch-03": "C",
            }[step_id]
            material = self.composition.material_by_id[material_id]
            source = (
                f'material_batch("{step_id.removeprefix("material-")}", '
                f"materials=[{_material_source(material)}])"
            )
        return {"composition_source": source, "intent_summary": "test"}


@pytest.mark.parametrize("passes", [True, False])
def test_run_long_form_publishes_only_after_required_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passes: bool
) -> None:
    target_path, manifest_path, records_path = _write_target(tmp_path)
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    def fake_staged_generation(
        *args,
        output_dir,
        style_prompt_target,
        require_inner_structure,
        require_naturalness,
        require_phrase_structure,
        require_voice_structure,
        **kwargs,
    ):
        assert style_prompt_target["axis_targets"]["density"]
        assert require_inner_structure is True
        assert require_naturalness is True
        assert require_phrase_structure is True
        assert require_voice_structure is True
        output_dir.mkdir(parents=True, exist_ok=True)
        source = composition_to_source(_composition())
        (output_dir / "plan.music.py").write_text(source, encoding="utf-8")
        (output_dir / "final.music.py").write_text(source, encoding="utf-8")
        return _composition()

    monkeypatch.setattr(
        "llm_musical_composer.long_form_loop.run_staged_generation",
        fake_staged_generation,
    )
    monkeypatch.setattr(
        "llm_musical_composer.long_form_loop.build_material_development_reference_profile_from_directory",
        lambda *args, **kwargs: {"metrics": {}},
    )
    monkeypatch.setattr(
        "llm_musical_composer.long_form_loop.evaluate_long_form_candidate",
        lambda *args, **kwargs: {"status": "pass" if passes else "fail"},
    )

    result = run_long_form(
        source_dir=source_dir,
        target_path=target_path,
        manifest_path=manifest_path,
        reference_records_path=records_path,
        schema_path=schema,
        output_root=tmp_path / "runs",
        run_id=f"run-{passes}",
        model="test-model",
    )

    assert result["status"] == ("pass" if passes else "fail")
    assert (Path(result["run_dir"]) / "staged/final.mid").is_file()
    assert (Path(result["run_dir"]) / "final.mid").is_file() is passes
    assert (Path(result["run_dir"]) / "verification.json").is_file()
    state = json.loads((Path(result["run_dir"]) / "run-state.json").read_text(encoding="utf-8"))
    assert state["status"] == ("completed" if passes else "failed")
    assert state["steps"]["publish-final"]["status"] == ("completed" if passes else "failed")


def test_run_long_form_records_generation_contract_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path, manifest_path, records_path = _write_target(tmp_path)
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    run_root = tmp_path / "runs"

    def reject_generation(*args, **kwargs):
        raise LongFormGenerationError("Ar: registral variation must preserve rhythm and contour")

    monkeypatch.setattr(
        "llm_musical_composer.long_form_loop.run_staged_generation", reject_generation
    )

    with pytest.raises(LongFormGenerationError, match="registral variation"):
        run_long_form(
            source_dir=source_dir,
            target_path=target_path,
            manifest_path=manifest_path,
            reference_records_path=records_path,
            schema_path=schema,
            output_root=run_root,
            run_id="failed-generation",
            model="test-model",
        )

    run_dir = run_root / "failed-generation"
    state = json.loads((run_dir / "run-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["steps"]["generate-candidate"]["status"] == "failed"
    assert state["steps"]["generate-candidate"]["outputs"] == {
        "error": "Ar: registral variation must preserve rhythm and contour",
        "error_type": "LongFormGenerationError",
    }
    assert not (run_dir / "final.mid").exists()


def test_run_long_form_snapshots_and_uses_preset_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path, manifest_path, records_path = _write_target(tmp_path)
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    response_dir = tmp_path / "responses"
    response_dir.mkdir()
    saved = {"composition_source": LONG_PARTS_SOURCE, "intent_summary": "saved"}
    (response_dir / "plan.json").write_text(json.dumps(saved), encoding="utf-8")
    plan_brief = tmp_path / "plan-brief.txt"
    plan_brief.write_text("use six unequal parts and an early return", encoding="utf-8")

    def fake_staged_generation(runner, *args, plan_prompt, output_dir, base_input_hashes, **kwargs):
        assert isinstance(runner, PresetResponseRunner)
        assert runner.run("plan", "ignored") == saved
        assert "use six unequal parts and an early return" in plan_prompt
        assert base_input_hashes["plan_brief"] == sha256_file(plan_brief)
        output_dir.mkdir(parents=True, exist_ok=True)
        source = composition_to_source(_composition())
        (output_dir / "plan.music.py").write_text(source, encoding="utf-8")
        (output_dir / "final.music.py").write_text(source, encoding="utf-8")
        return _composition()

    monkeypatch.setattr(
        "llm_musical_composer.long_form_loop.run_staged_generation",
        fake_staged_generation,
    )
    monkeypatch.setattr(
        "llm_musical_composer.long_form_loop.build_material_development_reference_profile_from_directory",
        lambda *args, **kwargs: {"metrics": {}},
    )
    monkeypatch.setattr(
        "llm_musical_composer.long_form_loop.evaluate_long_form_candidate",
        lambda *args, **kwargs: {"status": "pass"},
    )

    result = run_long_form(
        source_dir=source_dir,
        target_path=target_path,
        manifest_path=manifest_path,
        reference_records_path=records_path,
        schema_path=schema,
        output_root=tmp_path / "runs",
        run_id="preset-run",
        model="test-model",
        preset_response_dir=response_dir,
        plan_brief_path=plan_brief,
    )

    run_dir = Path(result["run_dir"])
    assert (run_dir / "inputs/preset-responses/plan.json").is_file()
    assert (run_dir / "inputs/plan-brief.txt").read_text(encoding="utf-8") == (
        "use six unequal parts and an early return"
    )
    spec = json.loads((run_dir / "run-spec.json").read_text(encoding="utf-8"))
    assert set(spec["preset_response_sha256"]) == {"plan"}
    assert "style_target.py" in spec["implementation_sha256"]
    assert spec["style_target_feasibility"]["status"] == "pass"
    assert spec["style_target_feasibility"]["checks"]["density_level"][
        "maximum_reachable_level"
    ] == pytest.approx(950 / 180)
    assert spec["plan_brief_sha256"] == sha256_file(plan_brief)


def test_plan_source_and_partial_response_directory_can_be_combined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path, manifest_path, records_path = _write_target(tmp_path)
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    plan = tmp_path / "plan.music.py"
    plan.write_text(LONG_PARTS_SOURCE, encoding="utf-8")
    response_dir = tmp_path / "responses"
    response_dir.mkdir()
    saved_batch = {
        "composition_source": 'material_batch("batch-01", materials=[])',
        "intent_summary": "saved batch",
    }
    (response_dir / "material-batch-01.json").write_text(json.dumps(saved_batch), encoding="utf-8")

    def fake_staged_generation(runner, *args, output_dir, **kwargs):
        assert isinstance(runner, PresetResponseRunner)
        assert isinstance(runner.delegate, PresetPlanRunner)
        assert runner.run("plan", "ignored")["composition_source"] == LONG_PARTS_SOURCE
        assert runner.run("material-batch-01", "ignored") == saved_batch
        output_dir.mkdir(parents=True, exist_ok=True)
        source = composition_to_source(_composition())
        (output_dir / "plan.music.py").write_text(source, encoding="utf-8")
        (output_dir / "final.music.py").write_text(source, encoding="utf-8")
        return _composition()

    monkeypatch.setattr(
        "llm_musical_composer.long_form_loop.run_staged_generation", fake_staged_generation
    )
    monkeypatch.setattr(
        "llm_musical_composer.long_form_loop.build_material_development_reference_profile_from_directory",
        lambda *args, **kwargs: {"metrics": {}},
    )
    monkeypatch.setattr(
        "llm_musical_composer.long_form_loop.evaluate_long_form_candidate",
        lambda *args, **kwargs: {"status": "pass"},
    )

    result = run_long_form(
        source_dir=tmp_path,
        target_path=target_path,
        manifest_path=manifest_path,
        reference_records_path=records_path,
        schema_path=schema,
        output_root=tmp_path / "runs",
        run_id="combined-preset-run",
        model="test-model",
        plan_source_path=plan,
        preset_response_dir=response_dir,
    )

    assert result["status"] == "pass"


def test_long_form_main_passes_cli_values_to_the_run(monkeypatch, capsys) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr("llm_musical_composer.long_form_loop.run_long_form", fake_run)

    result = main(
        [
            "--source-dir",
            "source",
            "--style-target",
            "target.json",
            "--style-manifest",
            "manifest.json",
            "--reference-records",
            "records.jsonl",
            "--schema",
            "schema.json",
            "--output-root",
            "runs",
            "--run-id",
            "test-run",
            "--model",
            "test-model",
            "--preset-responses",
            "saved-responses",
            "--plan-brief",
            "plan-brief.txt",
        ]
    )

    assert result == 0
    assert captured["run_id"] == "test-run"
    assert captured["model"] == "test-model"
    assert captured["preset_response_dir"] == Path("saved-responses")
    assert captured["plan_brief_path"] == Path("plan-brief.txt")
    assert '"status": "pass"' in capsys.readouterr().out
