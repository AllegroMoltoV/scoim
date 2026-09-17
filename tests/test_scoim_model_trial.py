import base64
import hashlib
import json
from pathlib import Path

import pytest
from test_scoim_staged_realization import QueueRunner, _approved_script, _complete_responses

from scoim.proposal import ProposalRun
from scoim.realization_generation import _saved_staged_runner_issues, create_model_trial
from scoim.realization_record import package_failed_staged_realization_record
from scoim.script_validation import script_content_sha256
from scoim.staged_realization import initialize_staged_realization
from scoim.trial_bundle import create_trial_bundle, replay_trial_bundle, verify_failed_trial_bundle
from scoim.validation import IssueCode, ValidationIssue, content_sha256


class _TimeoutRunner:
    def preflight(self) -> tuple[ValidationIssue, ...]:
        return ()

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        return ProposalRun(
            provider="fake",
            model="test-model",
            model_settings={"timeout_seconds": 1},
            started=True,
            terminal_state="timeout",
            raw_response=None,
            events=b'{"type":"thread.started"}\n',
            stderr=b"timed out",
            issues=(
                ValidationIssue(
                    IssueCode.RUNNER_TIMEOUT,
                    "runner timed out",
                    "/runner",
                ),
            ),
        )


class _NeverRunner:
    def preflight(self) -> tuple[ValidationIssue, ...]:
        raise AssertionError("invalid input must be rejected before runner preflight")

    def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
        raise AssertionError("invalid input must be rejected before a model call")


def _started_failure_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "bundle"
    result = create_model_trial(
        _approved_script(),
        _TimeoutRunner(),
        tmp_path / "run",
        bundle_dir,
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="timeout-staged",
    )
    assert result.persisted is True
    return bundle_dir


def _validated_script() -> dict[str, object]:
    legacy = _approved_script()
    return {
        "document_type": "script",
        "schema_version": "0.2.0",
        "document_id": legacy["document_id"],
        "revision": legacy["revision"],
        "status": "validated",
        "source_flow": {
            "document_id": "source-flow",
            "revision": 1,
            "content_sha256": "0" * 64,
        },
        "script": legacy["script"],
    }


def _composition_manifest(document: dict[str, object]) -> bytes:
    return json.dumps(
        {
            "bundle_type": "composition",
            "schema_version": 2,
            "composition_id": document["document_id"],
            "terminal_state": "succeeded",
            "validated_script_content_sha256": script_content_sha256(document),
            "approved_flow_content_sha256": "0" * 64,
            "files": {},
        },
        sort_keys=True,
    ).encode("utf-8")


@pytest.mark.parametrize("composition_manifest", (None, b"{"))
def test_validated_trial_rejects_an_unreadable_composition_manifest(
    tmp_path: Path,
    composition_manifest: bytes | None,
) -> None:
    result = create_trial_bundle(
        _validated_script(),
        {},
        tmp_path / "bundle",
        trial_id="missing-composition",
        composition_manifest=composition_manifest,
    )

    assert result.created is False
    assert result.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert result.issues[0].path == "/composition_manifest"
    assert not (tmp_path / "bundle").exists()


def test_validated_trial_rejects_a_composition_manifest_for_another_script(
    tmp_path: Path,
) -> None:
    document = _validated_script()
    manifest = json.loads(_composition_manifest(document))
    manifest["composition_id"] = "another-composition"

    result = create_trial_bundle(
        document,
        {},
        tmp_path / "bundle",
        trial_id="wrong-composition",
        composition_manifest=json.dumps(manifest).encode("utf-8"),
    )

    assert result.created is False
    assert result.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert result.issues[0].path == "/composition_manifest"
    assert not (tmp_path / "bundle").exists()


def test_create_staged_model_trial_accepts_a_validated_script_document(
    tmp_path: Path,
) -> None:
    document = _validated_script()
    composition_manifest = _composition_manifest(document)
    result = create_model_trial(
        document,
        QueueRunner(_complete_responses()),
        tmp_path / "run",
        tmp_path / "bundle",
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="validated-staged",
        composition_manifest=composition_manifest,
    )

    assert result.created is True, result.issues
    assert (tmp_path / "bundle" / "artifacts" / "final.mid").is_file()
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_type"] == "realization"
    assert manifest["schema_version"] == 4
    assert manifest["source_composition"] == {
        "manifest_sha256": hashlib.sha256(composition_manifest).hexdigest()
    }
    assert manifest["files"]["validated_script"]["path"] == "validated-script.json"
    assert manifest["files"]["source_composition_manifest"]["path"] == (
        "source-composition-manifest.json"
    )
    replayed = replay_trial_bundle(tmp_path / "bundle", tmp_path / "replay")
    assert replayed.replayed is True, replayed.issues


def test_failed_validated_trial_preserves_its_composition_lineage(tmp_path: Path) -> None:
    document = _validated_script()
    composition_manifest = _composition_manifest(document)

    result = create_model_trial(
        document,
        _TimeoutRunner(),
        tmp_path / "run",
        tmp_path / "bundle",
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="failed-validated-staged",
        composition_manifest=composition_manifest,
    )

    assert result.created is False
    assert result.persisted is True
    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_type"] == "realization"
    assert manifest["schema_version"] == 4
    assert manifest["source_composition"] == {
        "manifest_sha256": hashlib.sha256(composition_manifest).hexdigest()
    }
    assert manifest["files"]["validated_script"]["path"] == "validated-script.json"
    assert verify_failed_trial_bundle(tmp_path / "bundle") is None


def test_validated_trial_replay_rejects_a_changed_composition_manifest(
    tmp_path: Path,
) -> None:
    document = _validated_script()
    result = create_model_trial(
        document,
        QueueRunner(_complete_responses()),
        tmp_path / "run",
        tmp_path / "bundle",
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="changed-composition",
        composition_manifest=_composition_manifest(document),
    )
    assert result.created is True
    source_path = tmp_path / "bundle" / "source-composition-manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["approved_flow_content_sha256"] = "f" * 64
    source_path.write_text(json.dumps(source), encoding="utf-8")
    manifest_path = tmp_path / "bundle" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["source_composition_manifest"]["sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    replayed = replay_trial_bundle(tmp_path / "bundle", tmp_path / "replay")

    assert replayed.replayed is False
    assert replayed.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert replayed.issues[0].path == "/manifest/source_composition/manifest_sha256"


def test_create_staged_model_trial_builds_a_bundle_from_the_completed_workspace(
    tmp_path: Path,
) -> None:
    result = create_model_trial(
        _approved_script(),
        QueueRunner(_complete_responses()),
        tmp_path / "run",
        tmp_path / "bundle",
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="fixed-staged",
    )

    assert result.created is True
    assert result.persisted is True
    assert result.trial_path == tmp_path / "bundle"
    record = json.loads((tmp_path / "bundle" / "model-runs" / "realization.json").read_text())
    assert record["schema_version"] == 2
    assert record["final_workspace_record_sha256"]
    assert (tmp_path / "bundle" / "artifacts" / "final.mid").is_file()


def test_create_staged_model_trial_persists_a_finitely_failed_content_repair(
    tmp_path: Path,
) -> None:
    responses = _complete_responses()
    responses[4:5] = [{"accompaniments": []}, {"accompaniments": []}]
    bundle_dir = tmp_path / "failed-bundle"

    result = create_model_trial(
        _approved_script(),
        QueueRunner(responses),
        tmp_path / "failed-run",
        bundle_dir,
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="failed-content-repair",
    )

    assert result.created is False
    assert result.persisted is True
    assert result.trial_path == bundle_dir
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    record = json.loads((bundle_dir / "model-runs" / "realization.json").read_text())
    assert record["status"] == "failed"
    assert record["failed_stage"] == "accompaniment"
    assert any(path.endswith("attempt-002/validation.json") for path in record["files"])


def test_create_staged_model_trial_rejects_existing_bundle_before_creating_a_run(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    runner = QueueRunner(_complete_responses())

    result = create_model_trial(
        _approved_script(),
        runner,
        tmp_path / "run",
        bundle_dir,
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="fixed-staged",
    )

    assert result.persisted is False
    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert runner.calls == 0
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("run_relative", "bundle_relative"),
    [
        ("same", "same"),
        ("work", "work/bundle"),
        ("bundle/run", "bundle"),
    ],
)
def test_create_staged_model_trial_rejects_overlapping_run_and_bundle_paths(
    tmp_path: Path,
    run_relative: str,
    bundle_relative: str,
) -> None:
    runner = QueueRunner(_complete_responses())

    result = create_model_trial(
        _approved_script(),
        runner,
        tmp_path / run_relative,
        tmp_path / bundle_relative,
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="fixed-staged",
    )

    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert runner.calls == 0
    assert not (tmp_path / run_relative).exists()
    assert not (tmp_path / bundle_relative).exists()


def test_create_staged_model_trial_rejects_an_empty_trial_id_before_creating_a_run(
    tmp_path: Path,
) -> None:
    runner = QueueRunner(_complete_responses())

    result = create_model_trial(
        _approved_script(),
        runner,
        tmp_path / "run",
        tmp_path / "bundle",
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id=" ",
    )

    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert runner.calls == 0
    assert not (tmp_path / "run").exists()


def test_create_staged_model_trial_runs_optional_preflight_before_creating_a_run(
    tmp_path: Path,
) -> None:
    class UnavailableRunner:
        def preflight(self) -> tuple[ValidationIssue, ...]:
            return (
                ValidationIssue(
                    IssueCode.RUNNER_UNAVAILABLE,
                    "runner is unavailable",
                    "/runner",
                ),
            )

        def run(self, prompt: str, response_schema_path: Path):
            raise AssertionError("preflight failure must prevent a model call")

    result = create_model_trial(
        _approved_script(),
        UnavailableRunner(),
        tmp_path / "run",
        tmp_path / "bundle",
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="fixed-staged",
    )

    assert result.issues[0].code is IssueCode.RUNNER_UNAVAILABLE
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "bundle").exists()


def test_create_staged_model_trial_can_resume_after_a_later_preflight_failure(
    tmp_path: Path,
) -> None:
    class FailsBeforeSecondOperation(QueueRunner):
        def __init__(self) -> None:
            super().__init__(_complete_responses())
            self.preflights = 0

        def preflight(self) -> tuple[ValidationIssue, ...]:
            self.preflights += 1
            if self.preflights == 3:
                return (
                    ValidationIssue(
                        IssueCode.RUNNER_UNAUTHENTICATED,
                        "authentication expired",
                        "/runner",
                    ),
                )
            return ()

    run_dir = tmp_path / "run"
    bundle_dir = tmp_path / "bundle"
    first_runner = FailsBeforeSecondOperation()

    stopped = create_model_trial(
        _approved_script(),
        first_runner,
        run_dir,
        bundle_dir,
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="fixed-staged",
    )

    assert stopped.persisted is False
    assert stopped.issues[0].code is IssueCode.RUNNER_UNAUTHENTICATED
    assert run_dir.is_dir()
    assert not bundle_dir.exists()
    assert first_runner.calls == 1

    resumed = create_model_trial(
        _approved_script(),
        QueueRunner(_complete_responses()[1:]),
        run_dir,
        bundle_dir,
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="fixed-staged",
    )

    assert resumed.persisted is True
    assert resumed.created is True


def test_create_staged_model_trial_persists_a_verifiable_started_timeout(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    result = create_model_trial(
        _approved_script(),
        _TimeoutRunner(),
        tmp_path / "run",
        bundle_dir,
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="timeout-staged",
    )

    assert result.created is False
    assert result.persisted is True
    assert result.issues[0].code is IssueCode.RUNNER_TIMEOUT
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["realizer"]["realizer_version"] == 5
    record = json.loads((bundle_dir / "model-runs" / "realization.json").read_text())
    assert record["schema_version"] == 3
    assert record["status"] == "failed"
    assert verify_failed_trial_bundle(bundle_dir) is None


@pytest.mark.parametrize(
    "case",
    (
        "terminal_state",
        "manifest_response_hash",
        "realizer_version",
        "files_not_object",
        "missing_file_record",
        "invalid_file_record",
        "missing_file",
        "changed_file",
        "unrecorded_file",
        "different_script",
        "invalid_realization_record",
        "invalid_record_fields",
        "different_record_script",
        "invalid_record_lineage",
        "invalid_record_issue",
    ),
)
def test_started_failure_bundle_rejects_tampering(tmp_path: Path, case: str) -> None:
    bundle = _started_failure_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if case == "terminal_state":
        terminal_path = bundle / "terminal.json"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal["terminal_state"] = "completed"
        terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
        manifest["files"]["terminal"]["sha256"] = hashlib.sha256(
            terminal_path.read_bytes()
        ).hexdigest()
    elif case == "manifest_response_hash":
        manifest["frozen_response_sha256"] = "0" * 64
    elif case == "realizer_version":
        manifest["realizer"]["realizer_version"] = 4
    elif case == "files_not_object":
        manifest["files"] = []
    elif case == "missing_file_record":
        manifest["files"].pop("terminal")
    elif case == "invalid_file_record":
        manifest["files"]["terminal"] = {}
    elif case == "missing_file":
        manifest["files"]["terminal"]["path"] = "missing.json"
    elif case == "changed_file":
        script_path = bundle / "approved-script.json"
        document = json.loads(script_path.read_text(encoding="utf-8"))
        document["script"]["title"] = "changed"
        script_path.write_text(json.dumps(document), encoding="utf-8")
    elif case == "unrecorded_file":
        (bundle / "extra.txt").write_text("extra\n", encoding="utf-8")
    elif case == "different_script":
        script_path = bundle / "approved-script.json"
        document = json.loads(script_path.read_text(encoding="utf-8"))
        document["document_id"] = "different"
        script_path.write_text(json.dumps(document), encoding="utf-8")
        manifest["files"]["approved_script"]["sha256"] = hashlib.sha256(
            script_path.read_bytes()
        ).hexdigest()
    elif case == "invalid_realization_record":
        record_path = bundle / "model-runs" / "realization.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["issues"] = []
        record_path.write_text(json.dumps(record), encoding="utf-8")
        manifest["files"]["realization_model_run"]["sha256"] = hashlib.sha256(
            record_path.read_bytes()
        ).hexdigest()
    elif case in {
        "invalid_record_fields",
        "different_record_script",
        "invalid_record_lineage",
        "invalid_record_issue",
    }:
        record_path = bundle / "model-runs" / "realization.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if case == "invalid_record_fields":
            record["schema_version"] = 2
        elif case == "different_record_script":
            packaged = record["files"]["inputs/approved-script.json"]
            document = json.loads(base64.b64decode(packaged["content_base64"]))
            document["document_id"] = "different"
            content = json.dumps(document).encode("utf-8")
            packaged["content_base64"] = base64.b64encode(content).decode("ascii")
            packaged["sha256"] = hashlib.sha256(content).hexdigest()
        elif case == "invalid_record_lineage":
            record["model"] = "different-model"
        else:
            record["issues"][0]["extra"] = True
        record_path.write_text(json.dumps(record), encoding="utf-8")
        manifest["files"]["realization_model_run"]["sha256"] = hashlib.sha256(
            record_path.read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    issue = verify_failed_trial_bundle(bundle)

    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID


def test_started_failure_bundle_rejects_a_terminal_issue_different_from_its_run_record(
    tmp_path: Path,
) -> None:
    bundle = _started_failure_bundle(tmp_path)
    terminal_path = bundle / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["issues"][0]["message"] = "different failure"
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["terminal"]["sha256"] = hashlib.sha256(terminal_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    issue = verify_failed_trial_bundle(bundle)

    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID


@pytest.mark.parametrize(
    "case",
    ("missing_bundle", "malformed_manifest", "malformed_script", "malformed_terminal"),
)
def test_failed_bundle_verifier_rejects_unreadable_required_inputs(
    tmp_path: Path,
    case: str,
) -> None:
    if case == "missing_bundle":
        bundle = tmp_path / "missing"
    else:
        bundle = _started_failure_bundle(tmp_path)
        filename = {
            "malformed_manifest": "manifest.json",
            "malformed_script": "approved-script.json",
            "malformed_terminal": "terminal.json",
        }[case]
        (bundle / filename).write_text("{", encoding="utf-8")

    issue = verify_failed_trial_bundle(bundle)

    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID


@pytest.mark.parametrize(
    ("runner_value", "expected_code"),
    (
        ("{", IssueCode.MODEL_OUTPUT_INVALID),
        ("[]", IssueCode.MODEL_OUTPUT_INVALID),
        ({"started": False}, IssueCode.MODEL_OUTPUT_INVALID),
        (
            {
                "started": True,
                "terminal_state": "timeout",
                "issues": [{"code": "unknown", "message": "bad", "path": "/runner"}],
            },
            IssueCode.RUNNER_TIMEOUT,
        ),
        (
            {"started": True, "terminal_state": "failed", "issues": []},
            IssueCode.RUNNER_FAILED,
        ),
    ),
)
def test_saved_runner_issue_classification_has_a_typed_fallback(
    tmp_path: Path,
    runner_value: object,
    expected_code: IssueCode,
) -> None:
    runner_path = tmp_path / "stages" / "01" / "attempts" / "one" / "attempt-001" / "runner.json"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text(
        runner_value if isinstance(runner_value, str) else json.dumps(runner_value),
        encoding="utf-8",
    )

    issues = _saved_staged_runner_issues(tmp_path, RuntimeError("failed"))

    assert issues[0].code is expected_code


def test_failed_staged_record_reports_an_incomplete_run_without_creating_data(
    tmp_path: Path,
) -> None:
    record, issue = package_failed_staged_realization_record(
        tmp_path / "missing-run",
        (ValidationIssue(IssueCode.RUNNER_FAILED, "failed", "/runner"),),
    )

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.STORAGE_ERROR
    assert not (tmp_path / "missing-run").exists()


@pytest.mark.parametrize(
    ("case", "expected_code"),
    (
        ("invalid_document", IssueCode.SCHEMA_INVALID),
        ("draft", IssueCode.SEMANTIC_INVALID),
        ("unsupported_profile", IssueCode.UNSUPPORTED_PROFILE),
        ("missing_proposal", IssueCode.STORAGE_ERROR),
    ),
)
def test_create_staged_model_trial_rejects_invalid_inputs_before_runner_preflight(
    tmp_path: Path,
    case: str,
    expected_code: IssueCode,
) -> None:
    document = _approved_script()
    profile = "solo_piano_3m_v1"
    proposal_record_dir = None
    if case == "invalid_document":
        document.pop("schema_version")
    elif case == "draft":
        document["status"] = "draft"
        document["approval"] = None
    elif case == "unsupported_profile":
        profile = "unknown"
    else:
        proposal_record_dir = tmp_path / "missing-proposal"

    result = create_model_trial(
        document,
        _NeverRunner(),
        tmp_path / "run",
        tmp_path / "bundle",
        profile=profile,
        model="test-model",
        trial_id="invalid-input",
        proposal_record_dir=proposal_record_dir,
    )

    assert result.issues[0].code is expected_code
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "bundle").exists()


def test_create_staged_model_trial_reports_a_final_record_packaging_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    issue = ValidationIssue(IssueCode.STORAGE_ERROR, "cannot package", "/realization_record")
    monkeypatch.setattr(
        "scoim.realization_generation.package_staged_realization_record",
        lambda *args: (None, issue),
    )

    result = create_model_trial(
        _approved_script(),
        QueueRunner(_complete_responses()),
        tmp_path / "run",
        tmp_path / "bundle",
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="record-failure",
    )

    assert result.created is False
    assert result.persisted is False
    assert result.issues == (issue,)
    assert not (tmp_path / "bundle").exists()


def test_create_staged_model_trial_reports_a_failed_record_packaging_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    issue = ValidationIssue(IssueCode.STORAGE_ERROR, "cannot package", "/realization_record")
    monkeypatch.setattr(
        "scoim.realization_generation.package_failed_staged_realization_record",
        lambda *args: (None, issue),
    )

    result = create_model_trial(
        _approved_script(),
        _TimeoutRunner(),
        tmp_path / "run",
        tmp_path / "bundle",
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="failed-record-failure",
    )

    assert result.created is False
    assert result.persisted is False
    assert result.issues == (issue,)
    assert not (tmp_path / "bundle").exists()


def test_create_staged_model_trial_rejects_a_different_model_before_calling_it(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    initialize_staged_realization(
        run_dir,
        _approved_script(),
        profile="solo_piano_3m_v1",
        model="first-model",
        review_after={},
    )
    runner = QueueRunner(_complete_responses())

    result = create_model_trial(
        _approved_script(),
        runner,
        run_dir,
        tmp_path / "bundle",
        profile="solo_piano_3m_v1",
        model="different-model",
        trial_id="fixed-staged",
    )

    assert result.persisted is False
    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert runner.calls == 0


def test_create_staged_model_trial_checks_profile_compatibility_before_runner_preflight(
    tmp_path: Path,
) -> None:
    class NeverPreflightRunner(QueueRunner):
        def preflight(self) -> tuple[ValidationIssue, ...]:
            raise AssertionError("invalid profile input must be rejected first")

    document = _approved_script()
    document["script"]["performance_setup"]["target_duration_seconds"] = 60
    document["approval"]["content_sha256"] = content_sha256(document)
    runner = NeverPreflightRunner(_complete_responses())

    result = create_model_trial(
        document,
        runner,
        tmp_path / "run",
        tmp_path / "bundle",
        profile="solo_piano_3m_v1",
        model="test-model",
        trial_id="fixed-staged",
    )

    assert result.issues[0].code is IssueCode.UNSUPPORTED_PROFILE
    assert runner.calls == 0
