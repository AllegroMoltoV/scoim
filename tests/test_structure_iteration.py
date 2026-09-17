from __future__ import annotations

import json
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import mido
import pytest

from llm_musical_composer.music_dsl import parse_composition
from llm_musical_composer.structure_iteration import (
    StructureRunStatus,
    _reference_data,
    main,
    run_structure_iteration,
    validate_closing_revision,
)

BASE_SOURCE = """composition(
    title="base",
    form=[use("A"), use("B"), use("A"), use("B"), use("A"), use("B"), use("A"), use("B")],
    materials=[
        material("A", duration_ms=6000, notes=[
            note("a1", at_ms=0, duration_ms=1000, pitch=60, velocity=60),
            note("a2", at_ms=2500, duration_ms=800, pitch=64, velocity=70),
        ]),
        material("B", duration_ms=6000, notes=[
            note("b1", at_ms=100, duration_ms=900, pitch=55, velocity=75),
            note("b2", at_ms=3000, duration_ms=700, pitch=67, velocity=80),
        ], pedals=[pedal("bp1", at_ms=0, value=80), pedal("bp2", at_ms=5800, value=0)]),
    ],
)"""

CLOSING = """material("closing", duration_ms=6000, notes=[
    note("closing_n1", at_ms=0, duration_ms=1200, pitch=48, velocity=54),
    note("closing_n2", at_ms=2200, duration_ms=1600, pitch=60, velocity=48),
    note("closing_n3", at_ms=4000, duration_ms=1800, pitch=48, velocity=40),
], pedals=[pedal("closing_p1", at_ms=0, value=60), pedal("closing_p2", at_ms=5700, value=0)])"""

AFTER_SOURCE = BASE_SOURCE.replace(
    'use("A"), use("B")],', 'use("A"), use("B"), use("closing")],'
).replace("    ],\n)", f"        {CLOSING},\n    ],\n)")


class FakeRunner:
    def __init__(self, sources: list[str]) -> None:
        self.sources = deque(sources)
        self.calls = 0

    def run(self, prompt: str, response_path) -> dict[str, object]:
        self.calls += 1
        response = {"composition_source": self.sources.popleft(), "intent_summary": "test"}
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(json.dumps(response), encoding="utf-8")
        return response


def test_closing_revision_preserves_existing_form_and_materials() -> None:
    before = parse_composition(BASE_SOURCE)
    after = parse_composition(AFTER_SOURCE)

    assert validate_closing_revision(before, after) == []

    changed_existing = parse_composition(AFTER_SOURCE.replace("pitch=60", "pitch=61"))
    assert "material A changed" in validate_closing_revision(before, changed_existing)


def test_iteration_stops_before_model_when_closure_control_failed(tmp_path) -> None:
    runner = FakeRunner([AFTER_SOURCE])

    result = run_structure_iteration(
        BASE_SOURCE,
        {"metrics": {}},
        {"status": "fail"},
        runner,
        tmp_path,
        random_seed=3,
    )

    assert result.status is StructureRunStatus.CLOSURE_CONTROL_FAILED
    assert result.call_count == 0
    assert runner.calls == 0


def test_iteration_revises_once_and_creates_blind_listening_pair(tmp_path) -> None:
    runner = FakeRunner([AFTER_SOURCE])

    result = run_structure_iteration(
        BASE_SOURCE,
        {"metrics": {}},
        {"status": "pass"},
        runner,
        tmp_path,
        random_seed=3,
    )

    assert result.status is StructureRunStatus.COMPLETED
    assert result.call_count == 1
    assert (tmp_path / "listen" / "sample-X.mid").is_file()
    assert (tmp_path / "before-structure.json").is_file()
    assert (tmp_path / "after-structure.json").is_file()


def test_iteration_uses_at_most_one_repair(tmp_path) -> None:
    runner = FakeRunner(["invalid()", AFTER_SOURCE])

    result = run_structure_iteration(
        BASE_SOURCE,
        {"metrics": {}},
        {"status": "pass"},
        runner,
        tmp_path,
    )

    assert result.status is StructureRunStatus.COMPLETED
    assert result.call_count == 2
    assert runner.calls == 2


def test_iteration_records_failure_after_the_only_repair(tmp_path) -> None:
    runner = FakeRunner(["invalid()", "still_invalid()"])

    result = run_structure_iteration(
        BASE_SOURCE,
        {"metrics": {}},
        {"status": "pass"},
        runner,
        tmp_path,
    )

    assert result.status is StructureRunStatus.REVISION_FAILED
    assert result.call_count == 2
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["error"].startswith("DslError")


def test_revision_requires_a_string_response_even_after_repair(tmp_path) -> None:
    class MissingSourceRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, prompt, response_path):
            self.calls += 1
            return {"intent_summary": "missing"}

    runner = MissingSourceRunner()
    result = run_structure_iteration(
        BASE_SOURCE,
        {"metrics": {}},
        {"status": "pass"},
        runner,
        tmp_path,
    )

    assert result.status is StructureRunStatus.REVISION_FAILED
    assert result.call_count == 2
    assert "composition_source" in result.error


def test_revision_validation_reports_independent_scope_failures() -> None:
    before = parse_composition(BASE_SOURCE)
    valid = parse_composition(AFTER_SOURCE)

    assert "title changed" in validate_closing_revision(before, replace(valid, title="other"))
    assert "form must append closing exactly once" in validate_closing_revision(
        before, replace(valid, form=before.form)
    )
    assert "material set must add only closing" in validate_closing_revision(
        before, replace(valid, materials=valid.materials[:-1])
    )
    short_closing = replace(valid.materials[-1], duration_ms=3_000)
    short = replace(valid, materials=(*valid.materials[:-1], short_closing))
    issues = validate_closing_revision(before, short)
    assert "closing duration must be between 4000 and 8000 ms" in issues
    assert "expanded duration must be between 52000 and 56000 ms" in issues
    pedal_on = replace(
        valid.materials[-1], pedals=(replace(valid.materials[-1].pedals[0], value=100),)
    )
    assert "pedal must be off by the final note ending" in validate_closing_revision(
        before, replace(valid, materials=(*valid.materials[:-1], pedal_on))
    )


def test_reference_data_excludes_named_files_and_records_failures(tmp_path) -> None:
    composition = parse_composition(BASE_SOURCE)
    from llm_musical_composer.smf_render import render_composition

    render_composition(composition, tmp_path / "kept.mid")
    render_composition(composition, tmp_path / "rut.mid")
    (tmp_path / "broken.mid").write_bytes(b"broken")
    empty = mido.MidiFile(type=0, ticks_per_beat=500)
    empty.tracks.append(mido.MidiTrack())
    empty.save(tmp_path / "empty.mid")

    corpus, names, unable = _reference_data(tmp_path)

    assert len(corpus) == 1
    assert names == ["kept.mid"]
    assert {item["name"] for item in unable} == {"broken.mid", "empty.mid"}


@pytest.mark.parametrize(
    ("status", "expected"),
    [(StructureRunStatus.COMPLETED, 0), (StructureRunStatus.REVISION_FAILED, 2)],
)
def test_main_writes_reference_artifacts_and_returns_status(
    tmp_path, monkeypatch, status, expected
) -> None:
    baseline = tmp_path / "base.music.py"
    baseline.write_text(BASE_SOURCE, encoding="utf-8")
    notes = [
        SimpleNamespace(pitch=60, onset_ms=0, duration_ms=400, velocity=60),
        SimpleNamespace(pitch=64, onset_ms=1_000, duration_ms=500, velocity=70),
    ]
    monkeypatch.setattr(
        "llm_musical_composer.structure_iteration._reference_data",
        lambda path: ([notes], ["one.mid"], []),
    )
    monkeypatch.setattr(
        "llm_musical_composer.structure_iteration.evaluate_ending_discrimination",
        lambda corpus: {"status": "pass"},
    )
    monkeypatch.setattr(
        "llm_musical_composer.structure_iteration.CodexExecRunner", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        "llm_musical_composer.structure_iteration.run_structure_iteration",
        lambda *args, **kwargs: SimpleNamespace(status=status),
    )
    output = tmp_path / "out"

    result = main(
        [
            "--baseline-source",
            str(baseline),
            "--output-root",
            str(output),
            "--run-id",
            "test-run",
        ]
    )

    assert result == expected
    assert (output / "reference-profile.json").is_file()
    assert (output / "reference-controls.json").is_file()
