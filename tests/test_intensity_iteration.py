from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from llm_musical_composer.composition_ir import (
    Composition,
    Material,
    Note,
    Pedal,
    Phrase,
    TonicEnding,
    Use,
)
from llm_musical_composer.intensity_iteration import (
    IntensityIterationError,
    LocalIntensityTriplet,
    _evaluate_full_quality,
    apply_velocity_only_repair,
    build_local_intensity_triplet,
    evaluate_upper_delay_artifact,
    generate_intensity_material,
    main,
    repair_main,
    repair_material_velocity,
    repair_saved_local_intensity_iteration,
    run_local_intensity_iteration,
    select_intensity_test_material,
)
from llm_musical_composer.long_form_generation import _material_source
from llm_musical_composer.music_dsl import DslError
from llm_musical_composer.run_state import sha256_file
from tests.test_music_dsl import LONG_PARTS_SOURCE


class FakeRunner:
    def __init__(self, source: str) -> None:
        self.source = source
        self.calls: list[tuple[str, str, dict[str, str] | None]] = []

    def run(self, step_id, prompt, input_hashes=None):
        self.calls.append((step_id, prompt, input_hashes))
        return {"composition_source": self.source, "intent_summary": "test"}


def _material(
    *,
    material_id: str = "A",
    notes: tuple[Note, ...] | None = None,
    pedals: tuple[Pedal, ...] | None = None,
) -> Material:
    return Material(
        material_id,
        5_500,
        notes
        or (
            Note("A-l1", 0, 1_000, 48, 80, "lower"),
            Note("A-u1", 0, 500, 60, 80, "upper"),
            Note("A-u2", 800, 500, 62, 80, "upper"),
            Note("A-l2", 1_600, 1_000, 53, 80, "lower"),
            Note("A-u3", 2_400, 500, 64, 80, "upper"),
        ),
        pedals
        or (
            Pedal("A-p1", 0, 127),
            Pedal("A-p2", 5_500, 0),
        ),
    )


def _composition() -> Composition:
    material = _material()
    other = _material(material_id="B")
    return Composition(
        title="test",
        tonal_center=0,
        mode="major",
        ending=TonicEnding(1_000),
        form=(Use("A"), Use("B")),
        materials=(material, other),
    )


def _batch(material: Material, batch_id: str = "intensity-high") -> str:
    return f'material_batch("{batch_id}", materials=[{_material_source(material)}])'


def _target(value: float) -> dict[str, object]:
    return {
        "status": "reachable",
        "value": value,
        "targets": {
            "note_rate_hz": 1.2 if value > 0 else 0.7,
            "attack_rate_hz": 1.0 if value > 0 else 0.5,
            "velocity_level": 0.8 if value > 0 else 0.5,
        },
        "target_counts": {
            "note_count": 7 if value > 0 else 4,
            "attack_count": 6 if value > 0 else 3,
        },
        "holds": {"mean_active_polyphony": {"minimum": 0.5, "maximum": 1.5}},
        "acceptance": {},
    }


def test_generation_changes_only_one_material_and_freezes_pedals() -> None:
    composition = _composition()
    current = composition.material_by_id["A"]
    revised = replace(
        current,
        notes=(*current.notes, Note("A-u4", 3_200, 400, 64, 96, "upper")),
    )
    runner = FakeRunner(_batch(revised))

    result = generate_intensity_material(
        composition,
        runner,
        material_id="A",
        control_value=1.0,
        target=_target(1.0),
        prompt_template=(
            "{{batch_id}} {{control_value}} {{current_material}} {{target}} {{tonal_context}}"
        ),
        base_input_hashes={"base": "hash"},
    )

    assert result.material_by_id["A"] == revised
    assert result.material_by_id["B"] == composition.material_by_id["B"]
    assert runner.calls[0][0] == "intensity-high"
    assert runner.calls[0][2]["base"] == "hash"


@pytest.mark.parametrize(
    ("revised", "message"),
    [
        (replace(_material(), material_id="B"), "material ID"),
        (replace(_material(), duration_ms=5_400), "duration"),
        (
            replace(
                _material(),
                pedals=(Pedal("A-p1", 0, 127), Pedal("A-p2", 5_000, 0)),
            ),
            "pedals",
        ),
        (
            replace(
                _material(),
                notes=(*_material().notes, Note("A-new", 3_000, 400, 66, 80, "upper")),
            ),
            "pitch-class",
        ),
    ],
)
def test_generation_rejects_scope_and_harmony_changes(revised: Material, message: str) -> None:
    with pytest.raises(IntensityIterationError, match=message):
        generate_intensity_material(
            _composition(),
            FakeRunner(_batch(revised)),
            material_id="A",
            control_value=1.0,
            target=_target(1.0),
            prompt_template=(
                "{{batch_id}} {{control_value}} {{current_material}} {{target}} {{tonal_context}}"
            ),
        )


def test_generation_rejects_wrong_batch_unreachable_target_and_placeholder() -> None:
    with pytest.raises(IntensityIterationError, match="batch ID"):
        generate_intensity_material(
            _composition(),
            FakeRunner(_batch(_material(), batch_id="wrong")),
            material_id="A",
            control_value=1.0,
            target=_target(1.0),
            prompt_template=(
                "{{batch_id}} {{control_value}} {{current_material}} {{target}} {{tonal_context}}"
            ),
        )


@pytest.mark.parametrize("value", [0.0, float("nan"), True])
def test_generation_rejects_invalid_local_control_values(value: float) -> None:
    with pytest.raises(IntensityIterationError):
        generate_intensity_material(
            _composition(),
            FakeRunner("unused"),
            material_id="A",
            control_value=value,
            target=_target(1.0),
            prompt_template="unused",
        )


def test_generation_rejects_unknown_material_and_multiple_returned_materials() -> None:
    with pytest.raises(IntensityIterationError, match="unknown"):
        generate_intensity_material(
            _composition(),
            FakeRunner("unused"),
            material_id="missing",
            control_value=1.0,
            target=_target(1.0),
            prompt_template="unused",
        )
    serialized = _material_source(_material())
    source = f'material_batch("intensity-high", materials=[{serialized}, {serialized}])'
    with pytest.raises(IntensityIterationError, match="exactly one"):
        generate_intensity_material(
            _composition(),
            FakeRunner(source),
            material_id="A",
            control_value=1.0,
            target=_target(1.0),
            prompt_template=(
                "{{batch_id}} {{control_value}} {{current_material}} {{target}} {{tonal_context}}"
            ),
        )


@pytest.mark.parametrize(
    ("revised", "message"),
    [
        (replace(_material(), derived_from="source"), "derivation"),
        (
            replace(
                _material(),
                notes=(*_material().notes, Note("high", 3_000, 400, 72, 80, "upper")),
            ),
            "pitch range",
        ),
        (
            replace(
                _material(),
                notes=tuple(replace(note, voice="upper") for note in _material().notes),
            ),
            "voice texture",
        ),
    ],
)
def test_generation_rejects_more_fixed_contract_changes(revised: Material, message: str) -> None:
    with pytest.raises(IntensityIterationError, match=message):
        generate_intensity_material(
            _composition(),
            FakeRunner(_batch(revised)),
            material_id="A",
            control_value=1.0,
            target=_target(1.0),
            prompt_template=(
                "{{batch_id}} {{control_value}} {{current_material}} {{target}} {{tonal_context}}"
            ),
        )
    with pytest.raises(IntensityIterationError, match="reachable"):
        generate_intensity_material(
            _composition(),
            FakeRunner("unused"),
            material_id="A",
            control_value=1.0,
            target={"status": "unreachable"},
            prompt_template="unused",
        )
    with pytest.raises(IntensityIterationError, match="unresolved"):
        generate_intensity_material(
            _composition(),
            FakeRunner("unused"),
            material_id="A",
            control_value=1.0,
            target=_target(1.0),
            prompt_template="{{batch_id}} {{missing}}",
        )


def test_upper_delay_artifact_rejects_systematic_micro_delay() -> None:
    delayed = tuple(
        note
        for index in range(4)
        for note in (
            Note(f"l{index}", index * 500, 400, 48, 80, "lower"),
            Note(f"u{index}", index * 500 + 20, 300, 60, 80, "upper"),
        )
    )
    natural = tuple(
        note
        for index in range(4)
        for note in (
            Note(f"l{index}", index * 500, 400, 48, 80, "lower"),
            Note(f"u{index}", index * 500, 300, 60, 80, "upper"),
        )
    )

    assert evaluate_upper_delay_artifact(delayed)["status"] == "fail"
    assert evaluate_upper_delay_artifact(natural)["status"] == "pass"


def test_low_generation_uses_low_batch_and_rejects_micro_delayed_revision() -> None:
    composition = _composition()
    current = composition.material_by_id["A"]
    low_runner = FakeRunner(_batch(current, batch_id="intensity-low"))
    result = generate_intensity_material(
        composition,
        low_runner,
        material_id="A",
        control_value=-1.0,
        target=_target(-1.0),
        prompt_template=(
            "{{batch_id}} {{control_value}} {{current_material}} {{target}} {{tonal_context}}"
        ),
    )
    assert result.material_by_id["A"] == current
    assert low_runner.calls[0][0] == "intensity-low"

    delayed_pairs = tuple(
        note
        for index in range(4)
        for note in (
            Note(f"A-l{index}", index * 500, 400, 48, 80, "lower"),
            Note(f"A-u{index}", index * 500 + 20, 300, 60, 80, "upper"),
        )
    )
    lower_responses = tuple(
        Note(f"A-lr{index}", index * 500 + 250, 350, 48, 80, "lower") for index in range(4)
    )
    delayed_notes = (*delayed_pairs, *lower_responses)
    delayed = replace(current, notes=delayed_notes)
    with pytest.raises(IntensityIterationError, match="micro-delayed"):
        generate_intensity_material(
            composition,
            FakeRunner(_batch(delayed)),
            material_id="A",
            control_value=1.0,
            target=_target(1.0),
            prompt_template=(
                "{{batch_id}} {{control_value}} {{current_material}} {{target}} {{tonal_context}}"
            ),
        )


def test_velocity_repair_uses_one_offset_and_preserves_dynamic_differences() -> None:
    material = replace(
        _material(),
        notes=tuple(
            Note(
                f"n{index}",
                index * 200,
                150,
                48 if index % 2 == 0 else 60,
                velocity,
                "lower" if index % 2 == 0 else "upper",
            )
            for index, velocity in enumerate((112, 113, 114, 115, 116, 117, 118, 119))
        ),
    )
    target = {
        "status": "reachable",
        "targets": {"velocity_level": 0.91},
        "acceptance": {"velocity_level": {"minimum": 0.87, "maximum": 0.95}},
    }

    repaired, report = repair_material_velocity(material, target)

    assert report["status"] == "repaired"
    assert report["offset"] == -3
    assert [note.velocity for note in repaired.notes] == [
        note.velocity - 3 for note in material.notes
    ]


def test_velocity_repair_rejects_an_unreachable_coarse_bin() -> None:
    material = replace(
        _material(),
        notes=tuple(replace(note, velocity=112) for note in _material().notes),
    )
    target = {
        "status": "reachable",
        "targets": {"velocity_level": 0.93},
        "acceptance": {"velocity_level": {"minimum": 0.90, "maximum": 0.95}},
    }

    with pytest.raises(IntensityIterationError, match="offset"):
        repair_material_velocity(material, target)


@pytest.mark.parametrize(
    ("material", "target", "message"),
    [
        (_material(), {}, "target is invalid"),
        (
            _material(),
            {
                "status": "unreachable",
                "targets": {"velocity_level": 0.5},
                "acceptance": {"velocity_level": {"minimum": 0.4, "maximum": 0.6}},
            },
            "target is invalid",
        ),
        (
            _material(),
            {
                "status": "reachable",
                "targets": {"velocity_level": 0.5},
                "acceptance": {"velocity_level": {"minimum": 0.6, "maximum": 0.4}},
            },
            "target range is invalid",
        ),
        (
            replace(_material(), notes=()),
            {
                "status": "reachable",
                "targets": {"velocity_level": 0.5},
                "acceptance": {"velocity_level": {"minimum": 0.4, "maximum": 0.6}},
            },
            "requires notes",
        ),
    ],
)
def test_velocity_repair_rejects_invalid_contracts(
    material: Material, target: dict, message: str
) -> None:
    with pytest.raises(IntensityIterationError, match=message):
        repair_material_velocity(material, target)


def test_velocity_only_repair_does_not_repair_other_metric_failures() -> None:
    composition = _composition()
    compositions = {-1.0: composition, 0.0: composition, 1.0: composition}
    evaluation = {"status": "fail", "issues": ["note_rate_hz"]}

    result = apply_velocity_only_repair(compositions, {}, evaluation, material_id="A")

    assert result.compositions == compositions
    assert result.evaluation is evaluation


def test_velocity_only_repair_changes_only_the_failing_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composition = _composition()
    compositions = {-1.0: composition, 0.0: composition, 1.0: composition}
    targets = {
        value: {"acceptance": {"velocity_level": {"minimum": 0.7, "maximum": 0.9}}}
        for value in compositions
    }
    evaluation = {
        "status": "fail",
        "issues": ["velocity_level"],
        "candidates": {
            "-1.0": {"observables": {"velocity_level": 0.8}},
            "1.0": {"observables": {"velocity_level": 1.0}},
        },
    }
    revised = replace(
        composition.material_by_id["A"],
        notes=tuple(replace(note, velocity=64) for note in composition.material_by_id["A"].notes),
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.repair_material_velocity",
        lambda material, target: (revised, {"status": "repaired", "offset": -16}),
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.evaluate_material_intensity_triplet",
        lambda *args: {"status": "pass", "issues": []},
    )

    result = apply_velocity_only_repair(compositions, targets, evaluation, material_id="A")

    assert result.compositions[-1.0] is composition
    assert result.compositions[0.0] is composition
    assert result.compositions[1.0].material_by_id["A"] == revised
    assert result.repairs == {"1.0": {"status": "repaired", "offset": -16}}


def test_selector_skips_parents_transitions_climax_and_release() -> None:
    materials = tuple(_material(material_id=name) for name in ("A", "Av", "T", "G", "X"))
    materials = tuple(
        replace(material, derived_from="A") if material.material_id == "Av" else material
        for material in materials
    )
    composition = Composition(
        title="selection",
        form=(
            Use("A", role="opening"),
            Use("Av", role="return"),
            Use("T", role="transition"),
            Use("G", role="climax"),
            Use("X", role="contrast"),
        ),
        materials=materials,
    )

    assert select_intensity_test_material(composition) == "X"


def test_selector_reports_when_no_safe_local_material_exists() -> None:
    composition = Composition(
        title="none",
        form=(Use("A", role="transition"),),
        materials=(_material(),),
    )
    with pytest.raises(IntensityIterationError, match="no independent"):
        select_intensity_test_material(composition)


def test_prompt_loader_rejects_empty_and_missing_files(tmp_path: Path) -> None:
    from llm_musical_composer.intensity_iteration import load_prompt

    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(IntensityIterationError, match="empty"):
        load_prompt(empty)
    with pytest.raises(IntensityIterationError, match="unable"):
        load_prompt(tmp_path / "missing.md")


def test_triplet_builder_calls_only_low_and_high_from_the_same_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composition = _composition()
    generated: list[float] = []

    def fake_resolve(material, records, *, value, **kwargs):
        return {"status": "reachable", "value": value}

    def fake_generate(source, runner, *, control_value, **kwargs):
        generated.append(control_value)
        return source

    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.resolve_material_intensity_target",
        fake_resolve,
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.generate_intensity_material", fake_generate
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.evaluate_material_intensity_triplet",
        lambda *args: {"status": "pass"},
    )

    result = build_local_intensity_triplet(
        composition,
        FakeRunner("unused"),
        [],
        reference_name="ref.mid",
        material_id="A",
        prompt_template="prompt",
    )

    assert generated == [-1.0, 1.0]
    assert result.compositions[0.0] is composition
    assert result.evaluation["status"] == "pass"


def test_triplet_builder_rejects_unknown_material_and_unreachable_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(IntensityIterationError, match="unknown"):
        build_local_intensity_triplet(
            _composition(),
            FakeRunner("unused"),
            [],
            reference_name="ref.mid",
            material_id="missing",
            prompt_template="prompt",
        )

    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.resolve_material_intensity_target",
        lambda material, records, *, value, **kwargs: {
            "status": "unreachable" if value == 1 else "reachable",
            "value": value,
        },
    )
    with pytest.raises(IntensityIterationError, match="unreachable"):
        build_local_intensity_triplet(
            _composition(),
            FakeRunner("unused"),
            [],
            reference_name="ref.mid",
            material_id="A",
            prompt_template="prompt",
        )


def test_full_quality_reports_fixed_contract_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = _composition()
    changed_b = replace(baseline.material_by_id["B"], duration_ms=6_000)
    candidate = replace(
        baseline,
        title="changed",
        tonal_center=1,
        form=(Use("A"),),
        phrases=(Phrase("p1", "part-a", "transition", None, 0, 0),),
        materials=(baseline.material_by_id["A"], changed_b),
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.evaluate_sustain_profile",
        lambda materials: {"status": "fail", "issues": ["sustain"]},
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.evaluate_piano_texture",
        lambda composition: {"status": "fail", "issues": ["texture"]},
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.evaluate_variation_contracts",
        lambda composition: {"status": "fail", "issues": ["variation"]},
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.parse_composition",
        lambda *args, **kwargs: (_ for _ in ()).throw(DslError("bad")),
    )

    report = _evaluate_full_quality(baseline, candidate, material_id="A", max_notes=0)

    assert report["status"] == "fail"
    assert "title changed" in report["issues"]
    assert "structure changed" in report["issues"]
    assert "phrase plan changed" in report["issues"]
    assert "tonal context or ending changed" in report["issues"]
    assert "material B changed outside scope" in report["issues"]
    assert "duration changed" in report["issues"]
    assert "note count exceeds 0" in report["issues"]
    assert "sustain" in report["issues"]
    assert "composition source is invalid: bad" in report["issues"]


def test_full_quality_passes_an_unchanged_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    composition = _composition()
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.evaluate_sustain_profile",
        lambda materials: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.evaluate_piano_texture",
        lambda candidate: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.evaluate_variation_contracts",
        lambda candidate: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.parse_composition",
        lambda *args, **kwargs: composition,
    )

    assert (
        _evaluate_full_quality(composition, composition, material_id="A", max_notes=950)["status"]
        == "pass"
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.parse_composition",
        lambda *args, **kwargs: replace(composition, title="different"),
    )
    assert (
        "composition source round-trip changed the candidate"
        in _evaluate_full_quality(composition, composition, material_id="A", max_notes=950)[
            "issues"
        ]
    )


def test_run_publishes_triplet_only_after_all_automatic_gates_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_source = tmp_path / "base.music.py"
    base_source.write_text(LONG_PARTS_SOURCE, encoding="utf-8")
    records = tmp_path / "files.jsonl"
    records.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "pass",
                "outputs": {"files.jsonl": sha256_file(records)},
            }
        ),
        encoding="utf-8",
    )
    prompt = tmp_path / "prompt.md"
    prompt.write_text("prompt", encoding="utf-8")
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")

    class DummyRunner:
        def __init__(self, **kwargs) -> None:
            self.run_store = kwargs["run_store"]

    def fake_build(composition, *args, **kwargs):
        return LocalIntensityTriplet(
            {-1.0: composition, 0.0: composition, 1.0: composition},
            {-1.0: {}, 0.0: {}, 1.0: {}},
            {"status": "pass"},
        )

    monkeypatch.setattr("llm_musical_composer.intensity_iteration.CodexExecRunner", DummyRunner)
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.load_reference_records", lambda path: []
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.build_local_intensity_triplet", fake_build
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.select_intensity_test_material",
        lambda composition: "A",
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration._evaluate_full_quality",
        lambda *args, **kwargs: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.render_composition",
        lambda composition, path: Path(path).write_bytes(b"MThd"),
    )

    result = run_local_intensity_iteration(
        base_source_path=base_source,
        run_dir=tmp_path / "run",
        records_path=records,
        records_manifest_path=manifest,
        prompt_path=prompt,
        schema_path=schema,
        reference_name="reference.mid",
        model="test-model",
    )

    assert result["status"] == "pass"
    assert result["material_id"] == "A"
    assert len(result["candidates"]) == 3
    assert all(Path(item["midi_path"]).is_file() for item in result["candidates"].values())
    assert (tmp_path / "run/evaluation.json").is_file()


def test_run_keeps_failed_triplet_staged_and_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_source = tmp_path / "base.music.py"
    base_source.write_text(LONG_PARTS_SOURCE, encoding="utf-8")
    records = tmp_path / "files.jsonl"
    records.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"status": "pass", "outputs": {"files.jsonl": sha256_file(records)}}),
        encoding="utf-8",
    )
    prompt = tmp_path / "prompt.md"
    prompt.write_text("prompt", encoding="utf-8")
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")

    class DummyRunner:
        def __init__(self, **kwargs) -> None:
            pass

    def fake_build(composition, *args, **kwargs):
        return LocalIntensityTriplet(
            {-1.0: composition, 0.0: composition, 1.0: composition},
            {-1.0: {}, 0.0: {}, 1.0: {}},
            {"status": "fail"},
        )

    monkeypatch.setattr("llm_musical_composer.intensity_iteration.CodexExecRunner", DummyRunner)
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.load_reference_records", lambda path: []
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.build_local_intensity_triplet", fake_build
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.select_intensity_test_material",
        lambda composition: "A",
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration._evaluate_full_quality",
        lambda *args, **kwargs: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.render_composition",
        lambda composition, path: Path(path).write_bytes(b"MThd"),
    )

    result = run_local_intensity_iteration(
        base_source_path=base_source,
        run_dir=tmp_path / "run-fail",
        records_path=records,
        records_manifest_path=manifest,
        prompt_path=prompt,
        schema_path=schema,
        reference_name="reference.mid",
        model="test-model",
    )

    assert result["status"] == "fail"
    assert result["candidates"] == {}
    assert (tmp_path / "run-fail/staged/intensity-low.mid").is_file()
    assert not (tmp_path / "run-fail/candidates").exists()


def test_run_rejects_a_changed_reference_manifest_before_generation(
    tmp_path: Path,
) -> None:
    base_source = tmp_path / "base.music.py"
    base_source.write_text(LONG_PARTS_SOURCE, encoding="utf-8")
    records = tmp_path / "files.jsonl"
    records.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"status": "pass", "outputs": {"files.jsonl": "0" * 64}}),
        encoding="utf-8",
    )
    prompt = tmp_path / "prompt.md"
    prompt.write_text("prompt", encoding="utf-8")
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")

    with pytest.raises(IntensityIterationError, match="manifest"):
        run_local_intensity_iteration(
            base_source_path=base_source,
            run_dir=tmp_path / "run",
            records_path=records,
            records_manifest_path=manifest,
            prompt_path=prompt,
            schema_path=schema,
            reference_name="reference.mid",
            model="test-model",
        )


def test_saved_velocity_only_failure_is_repaired_without_an_external_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed_run = tmp_path / "failed"
    staged = failed_run / "staged"
    staged.mkdir(parents=True)
    for label in ("intensity-low", "intensity-zero", "intensity-high"):
        (staged / f"{label}.music.py").write_text(LONG_PARTS_SOURCE, encoding="utf-8")
    (failed_run / "evaluation.json").write_text(
        json.dumps(
            {
                "status": "fail",
                "reference": "ref.mid",
                "material_id": "A",
                "intensity": {
                    "status": "fail",
                    "issues": ["velocity_level"],
                    "candidates": {
                        "-1.0": {"observables": {"velocity_level": 0.8}},
                        "1.0": {"observables": {"velocity_level": 1.0}},
                    },
                },
                "targets": {
                    "-1.0": {"acceptance": {"velocity_level": {"minimum": 0.7, "maximum": 0.9}}},
                    "0.0": {},
                    "1.0": {"acceptance": {"velocity_level": {"minimum": 0.7, "maximum": 0.9}}},
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_apply(compositions, targets, evaluation, *, material_id):
        return LocalIntensityTriplet(
            compositions,
            targets,
            {"status": "pass", "issues": []},
            {"1.0": {"offset": -3}},
        )

    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.apply_velocity_only_repair", fake_apply
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration._evaluate_full_quality",
        lambda *args, **kwargs: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.render_composition",
        lambda composition, path: Path(path).write_bytes(b"MThd"),
    )

    result = repair_saved_local_intensity_iteration(
        failed_run=failed_run,
        run_dir=tmp_path / "repaired",
    )

    assert result["status"] == "pass"
    assert len(result["candidates"]) == 3
    assert all(Path(item["midi_path"]).is_file() for item in result["candidates"].values())


@pytest.mark.parametrize("manifest_value", ["not-json", json.dumps({"status": "fail"})])
def test_run_rejects_unreadable_or_failed_reference_manifest(
    tmp_path: Path, manifest_value: str
) -> None:
    base_source = tmp_path / "base.music.py"
    base_source.write_text(LONG_PARTS_SOURCE, encoding="utf-8")
    records = tmp_path / "files.jsonl"
    records.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(manifest_value, encoding="utf-8")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("prompt", encoding="utf-8")
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")

    with pytest.raises(IntensityIterationError, match="manifest"):
        run_local_intensity_iteration(
            base_source_path=base_source,
            run_dir=tmp_path / "run",
            records_path=records,
            records_manifest_path=manifest,
            prompt_path=prompt,
            schema_path=schema,
            reference_name="reference.mid",
            model="test-model",
        )


def test_main_passes_cli_values_to_the_run(monkeypatch, capsys) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "pass", "candidates": {}}

    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.run_local_intensity_iteration", fake_run
    )

    result = main(
        [
            "--base-source",
            "base.music.py",
            "--run-dir",
            "run",
            "--reference-records",
            "files.jsonl",
            "--reference-manifest",
            "manifest.json",
            "--prompt",
            "prompt.md",
            "--schema",
            "schema.json",
            "--reference",
            "ref.mid",
            "--material-id",
            "X",
            "--model",
            "test-model",
        ]
    )

    assert result == 0
    assert captured["reference_name"] == "ref.mid"
    assert captured["material_id"] == "X"
    assert captured["model"] == "test-model"
    assert '"status": "pass"' in capsys.readouterr().out

    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.run_local_intensity_iteration",
        lambda **kwargs: {"status": "fail", "candidates": {}},
    )
    assert (
        main(
            [
                "--base-source",
                "base.music.py",
                "--run-dir",
                "run-2",
                "--reference",
                "ref.mid",
            ]
        )
        == 2
    )


def test_repair_main_passes_cli_values_and_reports_failure(monkeypatch, capsys) -> None:
    captured = {}

    def fake_repair(**kwargs):
        captured.update(kwargs)
        return {"status": "pass", "candidates": {}}

    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.repair_saved_local_intensity_iteration",
        fake_repair,
    )

    result = repair_main(
        [
            "--failed-run",
            "failed-run",
            "--run-dir",
            "repaired-run",
            "--max-notes",
            "900",
        ]
    )

    assert result == 0
    assert captured == {
        "failed_run": Path("failed-run"),
        "run_dir": Path("repaired-run"),
        "max_notes": 900,
    }
    assert '"status": "pass"' in capsys.readouterr().out

    monkeypatch.setattr(
        "llm_musical_composer.intensity_iteration.repair_saved_local_intensity_iteration",
        lambda **kwargs: {"status": "fail", "candidates": {}},
    )
    assert (
        repair_main(
            [
                "--failed-run",
                "failed-run",
                "--run-dir",
                "repaired-run-2",
            ]
        )
        == 2
    )
