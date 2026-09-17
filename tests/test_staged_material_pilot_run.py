from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llm_musical_composer.pipeline_dsl import parse_score_spec
from llm_musical_composer.staged_material_pilot import (
    HarmonicDraft,
    HarmonicEventDraft,
    MelodyDraft,
    MelodyEventDraft,
    TextureDraft,
    TextureEventDraft,
    dump_harmonic_draft,
    dump_melody_draft,
    dump_texture_draft,
)
from llm_musical_composer.staged_material_pilot_run import (
    CASES,
    PROTOCOL_ID,
    create_default_pilot_runner,
    execute_prepared_case,
    execute_prepared_pilot,
    prepare_staged_material_pilot,
)


@dataclass
class FakeRunner:
    responses: dict[str, str]
    call_number: int = 0

    def run(
        self, step_id: str, prompt: str, input_hashes: dict[str, str] | None = None
    ) -> dict[str, object]:
        del prompt, input_hashes
        self.call_number += 1
        return {
            "composition_source": self.responses[step_id],
            "intent_summary": "test fixture",
        }


def _stage_responses(case_dir: Path) -> dict[str, str]:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    score = parse_score_spec((case_dir / "input-score-spec.dsl").read_text(encoding="utf-8"))
    target = next(
        material
        for material in score.materials
        if material.material_id == manifest["target_material_id"]
    )
    harmonies = HarmonicDraft(
        tuple(
            HarmonicEventDraft(
                item.at_units,
                item.duration_units,
                item.root_pitch_class,
                item.quality,
            )
            for item in target.harmonies
        )
    )
    melody_events: list[MelodyEventDraft] = []
    degrees = (0, 4, 7)
    foreground_base = 48 if target.foreground_voice == "lower" else 72
    for index, harmony in enumerate(target.harmonies):
        pitch_class = (harmony.root_pitch_class + degrees[index % len(degrees)]) % 12
        pitch = next(
            candidate
            for candidate in range(foreground_base, foreground_base + 18)
            if candidate % 12 == pitch_class
        )
        melody_events.append(
            MelodyEventDraft(
                harmony.at_units,
                min(score.divisions, harmony.duration_units),
                pitch,
            )
        )
    while len(melody_events) < 4 or len({item.pitch for item in melody_events}) < 3:
        harmony = target.harmonies[0]
        offset = len(melody_events)
        pitch_class = (harmony.root_pitch_class + degrees[offset % len(degrees)]) % 12
        pitch = next(
            candidate
            for candidate in range(foreground_base, foreground_base + 18)
            if candidate % 12 == pitch_class
        )
        melody_events.append(MelodyEventDraft(offset, 1, pitch))
    melody = MelodyDraft(target.foreground_voice or "upper", tuple(melody_events))
    zone = "high" if melody.foreground_voice == "lower" else "low"
    texture_events = tuple(
        TextureEventDraft(index, item.at_units, item.duration_units, degree, zone)
        for index, item in enumerate(target.harmonies)
        for degree in ("root", "fifth")
    )
    return {
        "harmony": dump_harmonic_draft(harmonies),
        "melody": dump_melody_draft(melody),
        "texture": dump_texture_draft(TextureDraft(texture_events)),
    }


def test_prepare_two_cases_without_leaking_old_target_events(tmp_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[1]
    output = tmp_path / "pilot"

    result = prepare_staged_material_pilot(workspace, output)

    assert result["status"] == "prepared"
    assert result["prepared_case_count"] == 2
    for case in result["cases"]:
        case_dir = output / case["case_id"]
        manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
        score = parse_score_spec((case_dir / "input-score-spec.dsl").read_text(encoding="utf-8"))
        target = next(
            material
            for material in score.materials
            if material.material_id == manifest["target_material_id"]
        )
        prompt = (case_dir / "prompt-harmony.txt").read_text(encoding="utf-8")
        assert manifest["status"] == "prepared"
        assert manifest["protocol_id"] == "staged-material-pilot-v3"
        assert PROTOCOL_ID == "staged-material-pilot-v3"
        assert all(note.event_id not in prompt for note in target.notes)
        assert all(harmony.harmony_id not in prompt for harmony in target.harmonies)


def test_default_runner_uses_an_isolated_working_directory(tmp_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[1]
    output = tmp_path / "pilot"
    prepare_staged_material_pilot(workspace, output)
    case_dir = output / "multiscale-v8-b-return"

    runner = create_default_pilot_runner(workspace, case_dir)

    assert runner.working_directory is not None
    assert not runner.working_directory.is_relative_to(output.resolve())


def test_execute_prepared_cases_with_three_distinct_stages(tmp_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[1]
    output = tmp_path / "pilot"
    prepare_staged_material_pilot(workspace, output)

    for case_dir in sorted(path for path in output.iterdir() if path.is_dir()):
        runner = FakeRunner(_stage_responses(case_dir))

        result = execute_prepared_case(case_dir, runner)

        assert result["status"] == "completed", result.get("error")
        assert result["passes"] is True
        assert result["confirmed_external_call_count"] == 3
        assert (case_dir / "outputs/final.mid").is_file()
        before = parse_score_spec(
            (case_dir / "input-score-spec.dsl").read_text(encoding="utf-8")
        )
        after = parse_score_spec(
            (case_dir / "combined-score-spec.dsl").read_text(encoding="utf-8")
        )
        target_id = str(result["target_material_id"])
        assert tuple(item for item in after.materials if item.material_id != target_id) == tuple(
            item for item in before.materials if item.material_id != target_id
        )
        before_target = next(item for item in before.materials if item.material_id == target_id)
        after_target = next(item for item in after.materials if item.material_id == target_id)
        assert (
            after_target.material_id,
            after_target.length_units,
            after_target.derived_from,
            after_target.directions,
        ) == (
            before_target.material_id,
            before_target.length_units,
            before_target.derived_from,
            before_target.directions,
        )


def test_execute_stops_after_first_invalid_stage(tmp_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[1]
    output = tmp_path / "pilot"
    prepare_staged_material_pilot(workspace, output)
    case_dir = output / "multiscale-v8-b-return"
    runner = FakeRunner(
        {
            "harmony": "harmonic_draft(events=[])",
            "melody": "melody_draft(foreground_voice='upper', events=[])",
            "texture": "texture_draft(events=[])",
        }
    )

    result = execute_prepared_case(case_dir, runner)

    assert result["status"] == "failed"
    assert result["confirmed_external_call_count"] == 1
    assert runner.call_number == 1

    repeated = execute_prepared_case(case_dir, runner)

    assert repeated == result
    assert runner.call_number == 1


def test_texture_feasibility_failure_persists_diagnostics_before_placement(
    tmp_path: Path,
) -> None:
    workspace = Path(__file__).resolve().parents[1]
    output = tmp_path / "pilot"
    prepare_staged_material_pilot(workspace, output)
    case_dir = output / "multiscale-v8-b-return"
    responses = _stage_responses(case_dir)
    responses["texture"] = responses["texture"].replace("'high'", "'bass'")

    result = execute_prepared_case(case_dir, FakeRunner(responses))

    assert result["status"] == "failed"
    assert result["confirmed_external_call_count"] == 3
    assert (case_dir / "response-texture.txt").is_file()
    assert (case_dir / "texture-draft.dsl").is_file()
    diagnostic = json.loads(
        (case_dir / "texture-feasibility.json").read_text(encoding="utf-8")
    )
    assert diagnostic["status"] == "declared_zone_infeasible"
    assert diagnostic["violations"]
    assert not (case_dir / "texture-placement.json").exists()

def test_pilot_continues_the_independent_case_after_one_failure(tmp_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[1]
    output = tmp_path / "pilot"
    prepare_staged_material_pilot(workspace, output)

    def runner_factory(case_dir: Path) -> FakeRunner:
        responses = _stage_responses(case_dir)
        if case_dir.name == "multiscale-v8-b-return":
            responses["harmony"] = "harmonic_draft(events=[])"
        return FakeRunner(responses)

    result = execute_prepared_pilot(workspace, output, runner_factory)

    assert result["status"] == "failed"
    assert result["passes"] is False
    assert result["confirmed_external_call_count"] == 4
    assert result["cases"][0]["status"] == "failed"
    assert result["cases"][1]["status"] == "completed"


def test_pilot_refuses_source_drift_before_calling_a_runner(tmp_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[1]
    output = tmp_path / "pilot"
    prepare_staged_material_pilot(workspace, output)
    case_dir = output / "multiscale-v8-b-return"
    run_spec_path = case_dir / "run-spec.json"
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    run_spec["inputs"]["runner"]["sha256"] = "0" * 64
    run_spec_path.write_text(json.dumps(run_spec), encoding="utf-8")
    factories_called: list[str] = []

    def runner_factory(path: Path) -> FakeRunner:
        factories_called.append(path.name)
        return FakeRunner(_stage_responses(path))

    result = execute_prepared_pilot(workspace, output, runner_factory)

    assert result["passes"] is False
    assert result["cases"][0]["status"] == "unable_to_investigate"
    assert result["cases"][0]["confirmed_external_call_count"] == 0
    assert factories_called == ["reference-v7-a2-material-7"]


def test_pilot_records_runner_factory_failure_and_continues(tmp_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[1]
    output = tmp_path / "pilot"
    prepare_staged_material_pilot(workspace, output)

    def runner_factory(case_dir: Path) -> FakeRunner:
        if case_dir.name == "multiscale-v8-b-return":
            raise OSError("isolated runner directory is unavailable")
        return FakeRunner(_stage_responses(case_dir))

    result = execute_prepared_pilot(workspace, output, runner_factory)

    assert result["passes"] is False
    assert result["cases"][0]["status"] == "unable_to_investigate"
    assert result["cases"][0]["error"]["type"] == "OSError"
    assert result["cases"][1]["status"] == "completed"
    assert result["confirmed_external_call_count"] == 3


def test_mock_success_outputs_are_deterministic_across_roots(tmp_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[1]
    roots = (tmp_path / "first", tmp_path / "second")
    hashes: list[tuple[str, str]] = []
    for output in roots:
        prepare_staged_material_pilot(workspace, output)

        def runner_factory(case_dir: Path) -> FakeRunner:
            return FakeRunner(_stage_responses(case_dir))

        result = execute_prepared_pilot(workspace, output, runner_factory)
        assert result["passes"] is True
        hashes.append(
            tuple(
                json.loads((output / case.case_id / "manifest.json").read_text("utf-8"))[
                    "smf_sha256"
                ]
                for case in CASES
            )
        )

    assert hashes[0] == hashes[1]
