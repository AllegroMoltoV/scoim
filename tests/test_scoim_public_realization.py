import json
from pathlib import Path

from test_scoim_script_compilation import FixedRunner, _approved_flow, _response
from test_scoim_staged_realization import QueueRunner, _complete_responses

import scoim
from scoim.composition_bundle import compose_flow
from scoim.proposal import ProposalRun
from scoim.public_realization import PublicRealizationResult, realize
from scoim.script_compilation import CompilationRequest
from scoim.validation import IssueCode


class NeverRunner:
    def preflight(self):
        raise AssertionError("An invalid input must be rejected before runner preflight")

    def run(self, prompt: str, response_schema_path: Path):
        raise AssertionError("An invalid input must be rejected before a model call")


class TimeoutRunner:
    def preflight(self) -> tuple:
        return ()

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        return ProposalRun(
            provider="fake",
            model="test-model",
            model_settings={},
            started=True,
            terminal_state="timeout",
            raw_response=None,
            events=None,
            stderr=b"timed out",
            issues=(),
        )


def test_public_api_exposes_the_realization_orchestrator() -> None:
    assert scoim.realize is realize
    assert scoim.PublicRealizationResult is PublicRealizationResult


def _fixed_aba_stage_responses() -> list[dict[str, object]]:
    responses = _complete_responses()
    responses[0]["contrast_descriptions"] = ["Open the register"]
    responses[3]["transition_melodies"] = responses[3]["transition_melodies"][:1]
    responses[5]["transition_accompaniments"] = responses[5]["transition_accompaniments"][:1]
    return [*responses[:7], responses[6]]


def test_realize_rejects_an_unknown_input_before_calling_the_runner(tmp_path: Path) -> None:
    source = tmp_path / "unknown.json"
    source.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    result = realize(
        source,
        tmp_path / "output",
        runner=NeverRunner(),
        model="test-model",
        trial_id="unknown",
    )

    assert result.succeeded is False
    assert result.persisted is False
    assert result.input_kind is None
    assert result.issues[0].code is IssueCode.UNSUPPORTED_SCHEMA_VERSION
    assert not (tmp_path / "output").exists()


def test_realize_rejects_a_non_object_input(tmp_path: Path) -> None:
    source = tmp_path / "list.json"
    source.write_text("[]", encoding="utf-8")

    result = realize(source, tmp_path / "output")

    assert result.issues[0].code is IssueCode.STORAGE_ERROR
    assert not (tmp_path / "output").exists()


def test_realize_rejects_an_existing_output_before_reading_input(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()

    result = realize(tmp_path / "missing.json", output)

    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT


def test_v1_realize_rejects_an_existing_v2_output_marker(tmp_path: Path) -> None:
    source = tmp_path / "approved-flow.json"
    source.write_text(json.dumps(_approved_flow()), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / "public-run.json").write_text("{}", encoding="utf-8")

    result = realize(
        source,
        output,
        runner=NeverRunner(),
        model="test-model",
        trial_id="trial-001",
        profile="solo_piano_3m_v1",
    )

    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT


def test_realize_rejects_an_unsupported_profile_before_reading_input(tmp_path: Path) -> None:
    result = realize(
        tmp_path / "missing.json",
        tmp_path / "output",
        profile="unsupported",
    )

    assert result.issues[0].code is IssueCode.UNSUPPORTED_PROFILE
    assert not (tmp_path / "output").exists()


def test_realize_requires_generation_arguments_for_an_approved_flow(tmp_path: Path) -> None:
    source = tmp_path / "approved-flow.json"
    source.write_text(json.dumps(_approved_flow()), encoding="utf-8")

    result = realize(source, tmp_path / "output")

    assert result.input_kind == "flow"
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID


def test_realize_rejects_an_invalid_flow_before_calling_the_runner(tmp_path: Path) -> None:
    document = _approved_flow()
    document.pop("title")
    source = tmp_path / "invalid-flow.json"
    source.write_text(json.dumps(document), encoding="utf-8")

    result = realize(
        source,
        tmp_path / "output",
        runner=NeverRunner(),
        model="test-model",
        trial_id="invalid-flow",
    )

    assert result.input_kind == "flow"
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID


def test_realize_rejects_a_draft_flow_before_calling_the_runner(tmp_path: Path) -> None:
    document = _approved_flow()
    document["status"] = "draft"
    document["approval"] = None
    source = tmp_path / "draft-flow.json"
    source.write_text(json.dumps(document), encoding="utf-8")

    result = realize(
        source,
        tmp_path / "output",
        runner=NeverRunner(),
        model="test-model",
        trial_id="draft-flow",
    )

    assert result.input_kind == "flow"
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert result.issues[0].path == "/status"


def test_realize_rejects_an_invalid_composition_id_before_calling_the_runner(
    tmp_path: Path,
) -> None:
    source = tmp_path / "approved-flow.json"
    source.write_text(json.dumps(_approved_flow()), encoding="utf-8")

    result = realize(
        source,
        tmp_path / "output",
        runner=NeverRunner(),
        model="test-model",
        trial_id="invalid-composition-id",
        composition_id=123,  # type: ignore[arg-type]
    )

    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert result.issues[0].path == "/composition_id"


def test_realize_preserves_a_failed_flow_compilation(tmp_path: Path) -> None:
    source = tmp_path / "approved-flow.json"
    source.write_text(json.dumps(_approved_flow()), encoding="utf-8")

    result = realize(
        source,
        tmp_path / "output",
        runner=TimeoutRunner(),
        model="test-model",
        trial_id="failed-compilation",
        profile="solo_piano_3m_v1",
    )

    assert result.succeeded is False
    assert result.persisted is True
    assert result.composition_bundle_path == Path("composition")
    assert result.trial_bundle_path is None
    assert (tmp_path / "output" / "composition" / "terminal.json").is_file()


def test_realize_creates_composition_and_trial_from_an_approved_flow(
    tmp_path: Path,
) -> None:
    source = tmp_path / "approved-flow.json"
    source.write_text(
        json.dumps(_approved_flow(), ensure_ascii=False),
        encoding="utf-8",
    )

    result = realize(
        source,
        tmp_path / "output",
        runner=QueueRunner([_response(), *_fixed_aba_stage_responses()]),
        model="test-model",
        trial_id="trial-001",
        profile="solo_piano_3m_v1",
    )

    assert result.succeeded is True, result.issues
    assert result.input_kind == "flow"
    assert result.composition_bundle_path == Path("composition")
    assert result.trial_bundle_path == Path("trial")
    assert result.artifacts == {
        "phase_04_melody": "trial/artifacts/phase-04-melody.mid",
        "phase_06_score": "trial/artifacts/phase-06-score.mid",
        "final_smf": "trial/artifacts/final.mid",
    }
    assert (tmp_path / "output" / "composition" / "validated-script.json").is_file()


def test_realize_creates_a_trial_from_a_verified_composition_bundle(tmp_path: Path) -> None:
    composition = tmp_path / "composition"
    composed = compose_flow(
        CompilationRequest(_approved_flow(), "composition-001", "solo_piano_3m_v1"),
        FixedRunner(_response()),
        composition,
    )
    assert composed.succeeded is True

    result = realize(
        composition,
        tmp_path / "output",
        runner=QueueRunner(_fixed_aba_stage_responses()),
        model="test-model",
        trial_id="trial-001",
        profile="solo_piano_3m_v1",
    )

    assert result.succeeded is True, result.issues
    assert result.input_kind == "composition"
    assert result.composition_bundle_path is None
    assert result.trial_bundle_path == Path("trial")
    assert result.artifacts == {
        "phase_04_melody": "trial/artifacts/phase-04-melody.mid",
        "phase_06_score": "trial/artifacts/phase-06-score.mid",
        "final_smf": "trial/artifacts/final.mid",
    }
    assert all((tmp_path / "output" / path).is_file() for path in result.artifacts.values())


def test_realize_requires_generation_arguments_for_a_composition_bundle(
    tmp_path: Path,
) -> None:
    composition = tmp_path / "composition"
    composed = compose_flow(
        CompilationRequest(_approved_flow(), "composition-001", "solo_piano_3m_v1"),
        FixedRunner(_response()),
        composition,
    )
    assert composed.succeeded is True

    result = realize(
        composition,
        tmp_path / "output",
        profile="solo_piano_3m_v1",
    )

    assert result.input_kind == "composition"
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID


def test_v1_composition_generation_requires_the_explicit_v1_profile(
    tmp_path: Path,
) -> None:
    composition = tmp_path / "composition"
    composed = compose_flow(
        CompilationRequest(_approved_flow(), "composition-001", "solo_piano_3m_v1"),
        FixedRunner(_response()),
        composition,
    )
    assert composed.succeeded is True

    result = realize(composition, tmp_path / "output")

    assert result.input_kind == "composition"
    assert result.issues[0].code is IssueCode.UNSUPPORTED_PROFILE


def test_realize_rejects_an_invalid_composition_bundle(tmp_path: Path) -> None:
    composition = tmp_path / "composition"
    composition.mkdir()
    (composition / "manifest.json").write_text(
        json.dumps({"bundle_type": "composition", "schema_version": 2}),
        encoding="utf-8",
    )

    result = realize(
        composition,
        tmp_path / "output",
        runner=NeverRunner(),
        model="test-model",
        trial_id="invalid-composition",
        profile="solo_piano_3m_v1",
    )

    assert result.input_kind == "composition"
    assert result.issues[0].code is IssueCode.STORAGE_ERROR


def test_realize_rejects_a_failed_composition_bundle(tmp_path: Path) -> None:
    composition = tmp_path / "composition"
    composed = compose_flow(
        CompilationRequest(_approved_flow(), "composition-001", "solo_piano_3m_v1"),
        TimeoutRunner(),
        composition,
    )
    assert composed.succeeded is False
    assert composed.persisted is True

    result = realize(
        composition,
        tmp_path / "output",
        runner=NeverRunner(),
        model="test-model",
        trial_id="failed-composition",
        profile="solo_piano_3m_v1",
    )

    assert result.input_kind == "composition"
    assert result.issues[0].code is IssueCode.SEMANTIC_INVALID
    assert not (tmp_path / "output").exists()


def test_realize_rejects_an_output_inside_the_composition_bundle(tmp_path: Path) -> None:
    composition = tmp_path / "composition"
    composed = compose_flow(
        CompilationRequest(_approved_flow(), "composition-001", "solo_piano_3m_v1"),
        FixedRunner(_response()),
        composition,
    )
    assert composed.succeeded is True

    result = realize(
        composition,
        composition / "output",
        runner=NeverRunner(),
        model="test-model",
        trial_id="nested-output",
        profile="solo_piano_3m_v1",
    )

    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert not (composition / "output").exists()


def test_realize_preserves_composition_and_failure_record_after_a_started_failure(
    tmp_path: Path,
) -> None:
    composition = tmp_path / "composition"
    composed = compose_flow(
        CompilationRequest(_approved_flow(), "composition-001", "solo_piano_3m_v1"),
        FixedRunner(_response()),
        composition,
    )
    assert composed.succeeded is True

    result = realize(
        composition,
        tmp_path / "output",
        runner=TimeoutRunner(),
        model="test-model",
        trial_id="failed-trial",
        profile="solo_piano_3m_v1",
    )

    assert result.succeeded is False
    assert result.persisted is True
    assert result.composition_bundle_path is None
    assert result.trial_bundle_path == Path("trial")
    assert (composition / "manifest.json").is_file()
    manifest = json.loads(
        (tmp_path / "output" / "trial" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["bundle_type"] == "realization"
    assert manifest["schema_version"] == 4


def test_realize_replays_a_trial_without_a_runner(tmp_path: Path) -> None:
    composition = tmp_path / "composition"
    compose_flow(
        CompilationRequest(_approved_flow(), "composition-001", "solo_piano_3m_v1"),
        FixedRunner(_response()),
        composition,
    )
    generated = realize(
        composition,
        tmp_path / "generated",
        runner=QueueRunner(_fixed_aba_stage_responses()),
        model="test-model",
        trial_id="trial-001",
        profile="solo_piano_3m_v1",
    )
    assert generated.succeeded is True

    replayed = realize(
        tmp_path / "generated" / "trial",
        tmp_path / "replayed",
    )

    assert replayed.succeeded is True, replayed.issues
    assert replayed.input_kind == "trial"
    assert replayed.replay_path == Path("artifacts")
    assert replayed.artifacts == {
        "phase_04_melody": "artifacts/phase-04-melody.mid",
        "phase_06_score": "artifacts/phase-06-score.mid",
        "final_smf": "artifacts/final.mid",
    }


def test_realize_rejects_generation_arguments_for_trial_replay(tmp_path: Path) -> None:
    composition = tmp_path / "composition"
    compose_flow(
        CompilationRequest(_approved_flow(), "composition-001", "solo_piano_3m_v1"),
        FixedRunner(_response()),
        composition,
    )
    generated = realize(
        composition,
        tmp_path / "generated",
        runner=QueueRunner(_fixed_aba_stage_responses()),
        model="test-model",
        trial_id="trial-001",
        profile="solo_piano_3m_v1",
    )
    assert generated.succeeded is True

    replayed = realize(
        tmp_path / "generated" / "trial",
        tmp_path / "replayed",
        runner=NeverRunner(),
        model="unused-model",
        trial_id="unused-trial",
    )

    assert replayed.succeeded is False
    assert replayed.issues[0].code is IssueCode.SCHEMA_INVALID
    assert not (tmp_path / "replayed").exists()
