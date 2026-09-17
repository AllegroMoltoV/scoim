import json
import shutil
from pathlib import Path

from test_scoim_script_compilation import FailureRunner, FixedRunner, _approved_flow, _response

from scoim.composition_bundle import compose_flow, verify_composition_bundle
from scoim.script_compilation import CompilationRequest


def test_success_bundle_is_atomic_self_contained_and_copyable(tmp_path: Path) -> None:
    destination = tmp_path / "composition-bundle"
    result = compose_flow(
        CompilationRequest(_approved_flow(), "composition-001", "solo_piano_3m_v1"),
        FixedRunner(_response()),
        destination,
    )

    assert result.persisted is True
    assert result.succeeded is True
    assert (destination / "validated-script.json").is_file()
    assert (destination / "model-runs" / "compilation.json").is_file()
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_type"] == "composition"
    assert manifest["schema_version"] == 2
    copied = tmp_path / "copied-bundle"
    shutil.copytree(destination, copied)
    assert verify_composition_bundle(copied).valid is True


def test_bundle_verification_detects_a_modified_fixed_file(tmp_path: Path) -> None:
    destination = tmp_path / "composition-bundle"
    compose_flow(
        CompilationRequest(_approved_flow(), "composition-001", "solo_piano_3m_v1"),
        FixedRunner(_response()),
        destination,
    )
    terminal_path = destination / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["state"] = "failed"
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")

    result = verify_composition_bundle(destination)

    assert result.valid is False


def test_failed_bundle_keeps_evidence_but_not_a_validated_script(tmp_path: Path) -> None:
    response = _response()
    response["scene_section_mappings"][0]["section_ids"] = ["contrast"]
    destination = tmp_path / "failed-bundle"

    result = compose_flow(
        CompilationRequest(_approved_flow(), "composition-002", "solo_piano_3m_v1"),
        FixedRunner([response, response]),
        destination,
    )

    assert result.persisted is True
    assert result.succeeded is False
    assert not (destination / "validated-script.json").exists()
    assert (
        destination
        / "model-runs"
        / "script-compilation"
        / "attempts"
        / "script-compilation"
        / "attempt-002"
    ).is_dir()
    assert verify_composition_bundle(destination).valid is True


def test_runner_failure_bundle_keeps_the_complete_model_attempt(tmp_path: Path) -> None:
    destination = tmp_path / "failed-bundle"

    result = compose_flow(
        CompilationRequest(_approved_flow(), "composition-002", "solo_piano_3m_v1"),
        FailureRunner(),
        destination,
    )

    attempt = (
        destination
        / "model-runs"
        / "script-compilation"
        / "attempts"
        / "script-compilation"
        / "attempt-001"
    )
    assert result.persisted is True
    assert result.succeeded is False
    assert (attempt / "request.json").is_file()
    assert (attempt / "prompt.md").is_file()
    assert (attempt / "stdout.jsonl").is_file()
    assert (attempt / "stderr.log").read_bytes() == b"failure"
    assert (attempt / "runner.json").is_file()
    assert (attempt / "terminal.json").is_file()
    assert verify_composition_bundle(destination).valid is True


def test_existing_bundle_destination_is_rejected_before_a_model_call(tmp_path: Path) -> None:
    destination = tmp_path / "composition-bundle"
    destination.mkdir()
    runner = FixedRunner(_response())

    result = compose_flow(
        CompilationRequest(_approved_flow(), "composition-003", "solo_piano_3m_v1"),
        runner,
        destination,
    )

    assert result.persisted is False
    assert runner.calls == 0
