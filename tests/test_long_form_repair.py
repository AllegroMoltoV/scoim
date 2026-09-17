from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from llm_musical_composer.composition_ir import Material, Note
from llm_musical_composer.long_form_generation import composition_to_source
from llm_musical_composer.long_form_repair import (
    LongFormRepairError,
    main,
    repair_material,
    run_repair,
)
from llm_musical_composer.music_dsl import THREE_MINUTE_POLICY, parse_composition
from tests.test_long_form_loop import _write_target
from tests.test_music_dsl import LONG_PARTS_SOURCE


class FakeRunner:
    def __init__(self, source: str) -> None:
        self.source = source
        self.calls = []

    def run(self, step_id, prompt, input_hashes=None):
        self.calls.append((step_id, prompt, input_hashes))
        return {"composition_source": self.source, "intent_summary": "test"}


def _composition():
    plan = parse_composition(
        LONG_PARTS_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    return replace(
        plan,
        materials=tuple(
            Material(
                material.material_id,
                material.duration_ms,
                (Note(f"{material.material_id}-1", 0, 500, 60, 70),),
            )
            for material in plan.materials
        ),
    )


def test_repair_changes_only_the_requested_material() -> None:
    composition = _composition()
    source = (
        'material_batch("repair-climax", materials=['
        'material("C", duration_ms=44000, notes=['
        'note("C-new", at_ms=0, duration_ms=800, pitch=72, velocity=110)], pedals=[])])'
    )
    runner = FakeRunner(source)

    repaired = repair_material(
        composition,
        runner,
        material_id="C",
        prompt_template="id={{material_id}} current={{current_material}} report={{evaluation}}",
        evaluation={"required": "increase"},
        base_input_hashes={"base": "hash"},
    )

    assert repaired.material_by_id["C"].notes[0].event_id == "C-new"
    assert repaired.material_by_id["A"] == composition.material_by_id["A"]
    assert runner.calls[0][0] == "repair-climax"
    assert runner.calls[0][2]["base"] == "hash"


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ('material_batch("wrong", materials=[])', "batch ID"),
        ('material_batch("repair-climax", materials=[])', "exactly one"),
        (
            'material_batch("repair-climax", materials=['
            'material("B", duration_ms=44000, notes=[], pedals=[])])',
            "material ID",
        ),
        (
            'material_batch("repair-climax", materials=['
            'material("C", duration_ms=43000, notes=[], pedals=[])])',
            "duration",
        ),
    ],
)
def test_repair_rejects_scope_and_duration_changes(source: str, message: str) -> None:
    with pytest.raises(LongFormRepairError, match=message):
        repair_material(
            _composition(),
            FakeRunner(source),
            material_id="C",
            prompt_template="{{material_id}} {{current_material}} {{evaluation}}",
            evaluation={},
        )


def test_repair_rejects_unknown_material_and_unresolved_prompt() -> None:
    with pytest.raises(LongFormRepairError, match="unknown"):
        repair_material(
            _composition(),
            FakeRunner("unused"),
            material_id="Z",
            prompt_template="unused",
            evaluation={},
        )
    with pytest.raises(LongFormRepairError, match="unresolved"):
        repair_material(
            _composition(),
            FakeRunner("unused"),
            material_id="C",
            prompt_template="{{material_id}} {{current_material}} {{evaluation}} {{missing}}",
            evaluation={},
        )


class _FakeRepairCodexRunner:
    def __init__(self, **kwargs) -> None:
        self.run_store = kwargs["run_store"]

    def run(self, step_id, prompt, input_hashes=None):
        source = (
            'material_batch("repair-climax", materials=['
            'material("C", duration_ms=44000, notes=['
            'note("C-new", at_ms=0, duration_ms=800, pitch=72, velocity=110)], pedals=[])])'
        )
        return {"composition_source": source, "intent_summary": "test"}


@pytest.mark.parametrize("passes", [True, False])
def test_run_repair_publishes_only_a_passing_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passes: bool
) -> None:
    base_run = tmp_path / "base"
    (base_run / "staged").mkdir(parents=True)
    (base_run / "staged/final.music.py").write_text(
        composition_to_source(_composition()), encoding="utf-8"
    )
    (base_run / "verification.json").write_text(
        json.dumps(
            {
                "macro_structure": {
                    "issues": ["climax failed"],
                    "climax": {"part_id": "P3"},
                    "parts": [
                        {
                            "part_id": "P3",
                            "role": "climax",
                            "features": {"note_density": 1},
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    target_path, manifest_path, records_path = _write_target(tmp_path)
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    monkeypatch.setattr(
        "llm_musical_composer.long_form_repair.CodexExecRunner",
        _FakeRepairCodexRunner,
    )
    monkeypatch.setattr(
        "llm_musical_composer.long_form_repair.build_material_development_reference_profile_from_directory",
        lambda *args, **kwargs: {"metrics": {}},
    )
    monkeypatch.setattr(
        "llm_musical_composer.long_form_repair.evaluate_long_form_candidate",
        lambda *args, **kwargs: {"status": "pass" if passes else "fail"},
    )

    result = run_repair(
        base_run=base_run,
        run_dir=tmp_path / f"repair-{passes}",
        source_dir=source_dir,
        target_path=target_path,
        manifest_path=manifest_path,
        reference_records_path=records_path,
        schema_path=schema,
        material_id="C",
        model="test-model",
    )

    assert result["status"] == ("pass" if passes else "fail")
    assert (Path(result["run_dir"]) / "staged/final.mid").is_file()
    assert (Path(result["run_dir"]) / "final.mid").is_file() is passes


def test_repair_main_passes_cli_values_to_the_run(monkeypatch, capsys) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr("llm_musical_composer.long_form_repair.run_repair", fake_run)

    result = main(
        [
            "--base-run",
            "base",
            "--run-dir",
            "repair",
            "--material-id",
            "C",
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
            "--model",
            "test-model",
        ]
    )

    assert result == 0
    assert captured["material_id"] == "C"
    assert captured["model"] == "test-model"
    assert '"status": "pass"' in capsys.readouterr().out
