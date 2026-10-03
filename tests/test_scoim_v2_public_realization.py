import json
import shutil
from pathlib import Path

import pytest
from test_scoim_script_0_3_compilation import (
    SequencedRunner,
    _approved_flow,
)
from test_scoim_script_0_4_compilation import _structure_response_0_4

import scoim.cli as cli_module
from llm_musical_composer.run_state import sha256_file, sha256_json
from scoim.cli import main
from scoim.phase8_bundle import verify_phase8_bundle
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


@pytest.fixture(autouse=True)
def _fixed_tonal_center(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scoim.phase3_realization.secrets.randbelow", lambda upper: 0)


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
            "mode": "major",
            "overall_harmonic_story": "主調を示して閉じる。",
            "total_score_units": 12,
            "section_harmonic_intents": [
                {
                    "harmonic_intent": "主調を示して閉じる。",
                    "connection_from_previous": "静かに始める。",
                }
            ],
        },
        {"harmonies": [{"duration_units": 12, "root_pitch_class": 0, "quality": "major"}]},
        {
            "foregrounds": [
                {"notes": [{"at_units": 0, "duration_units": 12, "pitch": 72, "voice": "upper"}]}
            ],
            "accompaniments": [],
        },
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


def test_saved_composition_is_preflighted_before_any_harmony_model_call(tmp_path):
    from dataclasses import asdict

    from test_scoim_phase4_realization import _document, _phase2_ledger

    from scoim.v2_public_realization import _realize_phases

    document = _document()
    document["script"]["script_element_variation_relations"]["theme-varied"]["target"] = {
        "type": "material_placement",
        "id": "support-return",
    }
    composition = tmp_path / "saved-composition"
    composition.mkdir()
    (composition / "validated-script.json").write_text(json.dumps(document), encoding="utf-8")
    (composition / "projection-ledger.json").write_text(
        json.dumps([asdict(entry) for entry in _phase2_ledger(document)]), encoding="utf-8"
    )
    runner = PublicV2Runner([])
    issues = _realize_phases(
        composition, tmp_path / "work", tmp_path / "trial", runner, "composition-001", "trial-001"
    )
    assert issues
    assert issues[0].code is IssueCode.UNREPRESENTABLE
    assert runner.calls == 0
    assert not (tmp_path / "work/phase3").exists()


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
        "phase_04_foreground": "realization-work/score/outputs/foreground-preview.mid",
        "phase_06_score": "realization-work/score/outputs/score-preview.mid",
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


@pytest.mark.parametrize("corruption", ["validation", "failed", "accepted-missing"])
def test_public_v2_realization_rejects_a_completed_phase_with_broken_model_records(
    tmp_path: Path,
    corruption: str,
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
    if corruption == "validation":
        path = next(
            (output / "realization-work/score/attempts").glob("*/attempt-*/validation.json")
        )
        path.unlink()
    elif corruption == "failed":
        path = next((output / "realization-work/phase3/attempts").glob("*/attempt-*/terminal.json"))
        value = json.loads(path.read_text(encoding="utf-8"))
        value["status"] = "failed"
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        for path in (output / "realization-work/phase3/events").glob("*/accepted.json"):
            path.unlink()
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


@pytest.mark.parametrize("keep_trial", [False, True])
def test_public_resume_rejects_a_different_completed_performance(tmp_path, keep_trial):
    composition = _composition_bundle(tmp_path)
    output = tmp_path / "output"
    donor = tmp_path / "donor"
    for directory, pitch in ((output, 72), (donor, 76)):
        responses = _generation_responses()
        responses[-1]["foregrounds"][0]["notes"][0]["pitch"] = pitch
        result = realize(
            composition,
            directory,
            runner=PublicV2Runner(responses),
            model="fixed",
            trial_id="trial-001",
        )
        assert result.succeeded, result.issues
    phase7 = output / "realization-work/phase7"
    phase7.rename(output / "original-phase7")
    shutil.copytree(donor / "realization-work/phase7", phase7)
    if not keep_trial:
        (output / "trial").rename(output / "original-trial")
    runner = PublicV2Runner([])
    resumed = realize(composition, output, runner=runner, model="fixed", trial_id="trial-001")
    assert not resumed.succeeded
    assert resumed.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert runner.calls == 0
    assert (output / "trial").exists() == keep_trial


@pytest.mark.parametrize("field", ["trial_id", "composition_id"])
def test_public_resume_rejects_a_foreign_valid_trial(tmp_path, field):
    from test_scoim_phase8_bundle import _refresh_bundle_inventory

    from llm_musical_composer.run_state import sha256_bytes

    composition = _composition_bundle(tmp_path)
    output = tmp_path / "output"
    initial = realize(
        composition,
        output,
        runner=PublicV2Runner(_generation_responses()),
        model="fixed",
        trial_id="trial-001",
    )
    assert initial.succeeded, initial.issues
    bundle = output / "trial"
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = "another-id"
    if field == "composition_id":
        path = bundle / "lineage/composition-manifest.json"
        composition_manifest = json.loads(path.read_text(encoding="utf-8"))
        composition_manifest[field] = "another-id"
        path.write_text(json.dumps(composition_manifest), encoding="utf-8")
        manifest["composition_manifest_sha256"] = sha256_bytes(path.read_bytes())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_bundle_inventory(bundle)
    checked = verify_phase8_bundle(bundle)
    assert checked.valid, checked.issues
    runner = PublicV2Runner([])
    resumed = realize(composition, output, runner=runner, model="fixed", trial_id="trial-001")
    assert not resumed.succeeded
    assert resumed.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert runner.calls == 0


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


@pytest.mark.parametrize("conflict", ["trial_id", "legacy_timing", "legacy_context"])
def test_public_v2_rejects_a_changed_run_contract_before_model_access(
    tmp_path: Path,
    conflict: str,
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
    runner = PublicV2Runner([])

    if conflict == "legacy_timing":
        marker_path = output / "public-run.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["schema_version"] = 1
        marker.pop("timing_contract")
        marker_path.write_text(json.dumps(marker), encoding="utf-8")

    if conflict == "legacy_context":
        marker_path = output / "public-run.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["schema_version"] = 2
        marker.pop("generation_context_contract")
        marker_path.write_text(json.dumps(marker), encoding="utf-8")

    changed = realize(
        composition,
        output,
        runner=runner,
        model="fixed",
        trial_id="trial-002" if conflict == "trial_id" else "trial-001",
    )

    assert changed.succeeded is False
    assert changed.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert runner.calls == 0
