import json
from pathlib import Path

import scoim.v2_public_run as public_run_module
from scoim.runner_identity import RunnerIdentity
from scoim.v2_public_run import PublicV2RunRequest, ensure_public_v2_run
from scoim.validation import IssueCode


def _request() -> PublicV2RunRequest:
    return PublicV2RunRequest(
        input_kind="flow",
        input_sha256="a" * 64,
        composition_id="composition-001",
        trial_id="trial-001",
        runner_identity=RunnerIdentity(
            "fake",
            "test-model",
            {"reasoning_effort": "medium"},
        ),
    )


def test_public_v2_run_initializes_its_identity_with_the_output_root(tmp_path: Path) -> None:
    output = tmp_path / "output"

    result = ensure_public_v2_run(output, _request())

    assert result.ready is True, result.issues
    assert result.resumed is False
    record = json.loads((output / "public-run.json").read_text(encoding="utf-8"))
    assert record == {
        "schema_version": 1,
        "operation": "scoim-public-v2-realization",
        "input_kind": "flow",
        "input_sha256": "a" * 64,
        "target_profile": "solo_piano_3m_v2",
        "composition_id": "composition-001",
        "trial_id": "trial-001",
        "runner_identity": {
            "provider": "fake",
            "model": "test-model",
            "model_settings": {"reasoning_effort": "medium"},
        },
    }


def test_public_v2_run_resumes_when_the_identity_matches(tmp_path: Path) -> None:
    output = tmp_path / "output"
    first = ensure_public_v2_run(output, _request())

    resumed = ensure_public_v2_run(output, _request())

    assert first.ready is True, first.issues
    assert resumed.ready is True, resumed.issues
    assert resumed.resumed is True
    assert resumed.output_dir == output.resolve()


def test_public_v2_run_rejects_a_changed_request_before_resuming(tmp_path: Path) -> None:
    output = tmp_path / "output"
    original = ensure_public_v2_run(output, _request())
    changed = PublicV2RunRequest(
        input_kind="flow",
        input_sha256="a" * 64,
        composition_id="composition-001",
        trial_id="trial-001",
        runner_identity=RunnerIdentity(
            "fake",
            "different-model",
            {"reasoning_effort": "medium"},
        ),
    )

    result = ensure_public_v2_run(output, changed)

    assert original.ready is True, original.issues
    assert result.ready is False
    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT


def test_public_v2_run_rejects_invalid_identity_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    invalid = PublicV2RunRequest(
        input_kind="flow",
        input_sha256="a" * 64,
        composition_id="composition-001",
        trial_id=" ",
        runner_identity=RunnerIdentity("fake", "test-model", {}),
    )

    result = ensure_public_v2_run(output, invalid)

    assert result.ready is False
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert not output.exists()


def test_public_v2_run_leaves_no_output_when_atomic_initialization_fails(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "output"

    def fail_write(path: Path, value: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(public_run_module, "atomic_write_json", fail_write)

    result = ensure_public_v2_run(output, _request())

    assert result.ready is False
    assert result.issues[0].code is IssueCode.STORAGE_ERROR
    assert not output.exists()
