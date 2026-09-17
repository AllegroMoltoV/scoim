import json
from pathlib import Path

from test_scoim_script_0_3_compilation import (
    SequencedRunner,
    _approved_flow,
)
from test_scoim_script_0_4_compilation import _structure_response_0_4

import scoim.cli as cli_module
from llm_musical_composer.run_state import sha256_file, sha256_json
from scoim.cli import main
from scoim.proposal import ProposalRun
from scoim.public_realization import realize
from scoim.runner_identity import RunnerIdentity
from scoim.script_0_4_compilation import (
    Script04CompilationRequest,
    compile_script_0_4,
)
from scoim.v2_composition_bundle import create_v2_composition_bundle
from scoim.v2_public_run import PublicV2RunRequest, ensure_public_v2_run
from scoim.validation import IssueCode


class PublicV2Runner:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.calls = 0

    def identity(self) -> RunnerIdentity:
        return RunnerIdentity("fixed", "fixed", {})

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        response = self.responses[self.calls]
        self.calls += 1
        return ProposalRun(
            provider="fixed",
            model="fixed",
            model_settings={},
            started=True,
            terminal_state="completed",
            raw_response=json.dumps(response, ensure_ascii=False).encode("utf-8"),
            events=b"",
            stderr=b"",
            issues=(),
        )


def _generation_responses() -> list[dict[str, object]]:
    return [
        {
            "tonal_center": 0,
            "mode": "major",
            "overall_harmonic_story": "主調を示して閉じる。",
            "section_harmonic_intents": [
                {
                    "harmonic_intent": "主調を示して閉じる。",
                    "connection_from_previous": "静かに始める。",
                }
            ],
        },
        {"harmonies": [{"duration_units": 12, "root_pitch_class": 0, "quality": "major"}]},
        {"notes": [{"at_units": 0, "duration_units": 12, "pitch": 72, "voice": "upper"}]},
    ]


def _composition_bundle(tmp_path: Path) -> Path:
    relations = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [],
        "performance_directions": [],
    }
    phase2 = tmp_path / "phase2"
    compiled = compile_script_0_4(
        Script04CompilationRequest(_approved_flow(), "composition-001"),
        SequencedRunner([_structure_response_0_4(), relations]),
        phase2,
    )
    assert compiled.compiled is True, compiled.issues
    composition = tmp_path / "composition"
    created = create_v2_composition_bundle(phase2, composition)
    assert created.created is True, created.issues
    return composition


def test_public_realize_runs_v2_composition_through_the_final_artifacts(
    tmp_path: Path,
) -> None:
    composition = _composition_bundle(tmp_path)
    runner = PublicV2Runner(_generation_responses())

    result = realize(
        composition,
        tmp_path / "output",
        runner=runner,
        model="fixed",
        trial_id="trial-001",
    )

    assert result.succeeded is True, result.issues
    assert result.input_kind == "composition"
    assert result.trial_bundle_path == Path("trial")
    assert result.artifacts == {
        "phase_04_foreground": "realization-work/phase4/outputs/foreground-preview.mid",
        "phase_06_score": "realization-work/phase6/outputs/score-preview.mid",
        "final_musicxml": "trial/artifacts/score.musicxml",
        "final_smf": "trial/artifacts/final.mid",
    }


def test_public_realize_runs_an_approved_flow_through_the_v2_default(
    tmp_path: Path,
) -> None:
    source = tmp_path / "flow.json"
    source.write_text(json.dumps(_approved_flow(), ensure_ascii=False), encoding="utf-8")
    relations = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [],
        "performance_directions": [],
    }
    runner = PublicV2Runner([_structure_response_0_4(), relations, *_generation_responses()])

    result = realize(
        source,
        tmp_path / "output",
        runner=runner,
        model="fixed",
        trial_id="trial-001",
    )

    assert result.succeeded is True, result.issues
    assert result.input_kind == "flow"
    assert result.composition_bundle_path == Path("composition")
    assert result.trial_bundle_path == Path("trial")
    assert runner.calls == 5
    manifest = json.loads(
        (tmp_path / "output" / "composition" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 3


def test_public_v2_realization_resumes_without_repeating_completed_model_calls(
    tmp_path: Path,
) -> None:
    composition = _composition_bundle(tmp_path)
    output = tmp_path / "output"
    first_runner = PublicV2Runner(_generation_responses())
    first = realize(
        composition,
        output,
        runner=first_runner,
        model="fixed",
        trial_id="trial-001",
    )
    resumed_runner = PublicV2Runner([])

    resumed = realize(
        composition,
        output,
        runner=resumed_runner,
        model="fixed",
        trial_id="trial-001",
    )

    assert first.succeeded is True, first.issues
    assert resumed.succeeded is True, resumed.issues
    assert resumed_runner.calls == 0


def test_public_v2_realization_rejects_a_completed_phase_with_broken_model_records(
    tmp_path: Path,
) -> None:
    composition = _composition_bundle(tmp_path)
    output = tmp_path / "output"
    first = realize(
        composition,
        output,
        runner=PublicV2Runner(_generation_responses()),
        model="fixed",
        trial_id="trial-001",
    )
    assert first.succeeded is True, first.issues
    validation_path = next(
        (output / "realization-work" / "phase4" / "attempts").glob("*/attempt-*/validation.json")
    )
    validation_path.unlink()
    resumed_runner = PublicV2Runner([])

    resumed = realize(
        composition,
        output,
        runner=resumed_runner,
        model="fixed",
        trial_id="trial-001",
    )

    assert resumed.succeeded is False
    assert resumed.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert resumed_runner.calls == 0


def test_public_v2_flow_resumes_after_a_completed_phase2_operation(tmp_path: Path) -> None:
    document = _approved_flow()
    source = tmp_path / "flow.json"
    source.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "output"
    identity = RunnerIdentity("fixed", "fixed", {})
    initialized = ensure_public_v2_run(
        output,
        PublicV2RunRequest(
            input_kind="flow",
            input_sha256=sha256_json(document),
            composition_id="flow-001",
            trial_id="trial-001",
            runner_identity=identity,
        ),
    )
    assert initialized.ready is True
    paused = compile_script_0_4(
        Script04CompilationRequest(document, "flow-001"),
        SequencedRunner([_structure_response_0_4()]),
        output / "realization-work" / "phase2",
        max_new_operations=1,
    )
    assert paused.outcome == "paused"
    relations = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [],
        "performance_directions": [],
    }
    runner = PublicV2Runner([relations, *_generation_responses()])

    result = realize(
        source,
        output,
        runner=runner,
        model="fixed",
        trial_id="trial-001",
    )

    assert result.succeeded is True, result.issues
    assert runner.calls == 4


def test_public_realize_replays_a_v2_trial_without_a_runner(tmp_path: Path) -> None:
    composition = _composition_bundle(tmp_path)
    generated = realize(
        composition,
        tmp_path / "generated",
        runner=PublicV2Runner(_generation_responses()),
        model="fixed",
        trial_id="trial-001",
    )
    assert generated.succeeded is True, generated.issues

    replayed = realize(tmp_path / "generated" / "trial", tmp_path / "replayed")

    assert replayed.succeeded is True, replayed.issues
    assert replayed.input_kind == "trial"
    assert replayed.artifacts == {
        "final_musicxml": "artifacts/score.musicxml",
        "final_smf": "artifacts/final.mid",
    }


def test_flow_and_its_v2_composition_bundle_produce_the_same_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "flow.json"
    source.write_text(json.dumps(_approved_flow(), ensure_ascii=False), encoding="utf-8")
    relations = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [],
        "performance_directions": [],
    }
    from_flow = realize(
        source,
        tmp_path / "from-flow",
        runner=PublicV2Runner([_structure_response_0_4(), relations, *_generation_responses()]),
        model="fixed",
        trial_id="trial-001",
    )
    assert from_flow.succeeded is True, from_flow.issues

    from_composition = realize(
        tmp_path / "from-flow" / "composition",
        tmp_path / "from-composition",
        runner=PublicV2Runner(_generation_responses()),
        model="fixed",
        trial_id="trial-001",
    )

    assert from_composition.succeeded is True, from_composition.issues
    for relative in ("score.musicxml", "final.mid"):
        assert sha256_file(tmp_path / "from-flow" / "trial" / "artifacts" / relative) == (
            sha256_file(tmp_path / "from-composition" / "trial" / "artifacts" / relative)
        )


def test_realize_cli_uses_v2_for_new_generation_by_default(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "flow.json"
    source.write_text(json.dumps(_approved_flow(), ensure_ascii=False), encoding="utf-8")
    relations = {
        "outcome": "complete",
        "structure_insufficient_reason": None,
        "variation_relations": [],
        "material_placement_transitions": [],
        "performance_directions": [],
    }
    runner = PublicV2Runner([_structure_response_0_4(), relations, *_generation_responses()])
    monkeypatch.setattr(cli_module, "CodexStructuredRunner", lambda **kwargs: runner)
    output = tmp_path / "output"

    exit_code = main(
        [
            "realize",
            str(source),
            "--model",
            "fixed",
            "--trial-id",
            "trial-001",
            "--output",
            str(output),
        ]
    )

    response = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert response["succeeded"] is True
    manifest = json.loads((output / "composition" / "manifest.json").read_text("utf-8"))
    assert manifest["target_profile"] == "solo_piano_3m_v2"


def test_public_v2_rejects_a_changed_trial_id_before_model_access(tmp_path: Path) -> None:
    composition = _composition_bundle(tmp_path)
    output = tmp_path / "output"
    first = realize(
        composition,
        output,
        runner=PublicV2Runner(_generation_responses()),
        model="fixed",
        trial_id="trial-001",
    )
    assert first.succeeded is True, first.issues
    runner = PublicV2Runner([])

    changed = realize(
        composition,
        output,
        runner=runner,
        model="fixed",
        trial_id="trial-002",
    )

    assert changed.succeeded is False
    assert changed.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert runner.calls == 0
