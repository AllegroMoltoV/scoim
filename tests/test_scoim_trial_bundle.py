import base64
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_scoim_realization_record import _completed_run
from test_scoim_staged_realization import _approved_script

import scoim.trial_bundle as trial_bundle_module
from scoim.operations import apply_patch, approve
from scoim.realization_record import package_staged_realization_record
from scoim.trial_bundle import create_trial_bundle, replay_trial_bundle
from scoim.validation import IssueCode, content_sha256

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "scoim" / "fixed-aba"
_NESTED_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "scoim" / "nested-aba"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((_FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _proposal_record(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    document = _fixture("approved-script.json")
    draft = deepcopy(document)
    draft["status"] = "draft"
    draft["approval"] = None
    draft["script"]["requirements"] = {}
    proposal = tmp_path / "proposal"
    proposal.mkdir()
    files = {
        "normalized-draft.json": (
            json.dumps(draft, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "prompt.md": b"test prompt\n",
        "request.json": b"{}\n",
        "response-schema.json": b"{}\n",
        "validation.json": b"{}\n",
    }
    for name, data in files.items():
        (proposal / name).write_bytes(data)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "runner": {
            "provider": "fake",
            "model": "test-model",
            "model_settings": {},
            "terminal_state": "completed",
        },
        "draft": {
            "document_id": draft["document_id"],
            "revision": draft["revision"],
            "content_sha256": content_sha256(draft),
        },
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    (proposal / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return proposal, document


def test_package_proposal_record_rejects_a_missing_directory(tmp_path: Path) -> None:
    record, issue = trial_bundle_module.package_proposal_record(
        tmp_path / "missing", _fixture("approved-script.json")
    )

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.STORAGE_ERROR
    assert issue.path == "/proposal_record_dir"


def test_package_proposal_record_rejects_a_missing_manifest(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal"
    proposal.mkdir()

    record, issue = trial_bundle_module.package_proposal_record(
        proposal, _fixture("approved-script.json")
    )

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/proposal_record/manifest.json"


def test_package_proposal_record_rejects_unknown_manifest_fields(tmp_path: Path) -> None:
    proposal, document = _proposal_record(tmp_path)
    manifest_path = proposal / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    record, issue = trial_bundle_module.package_proposal_record(proposal, document)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/proposal_record/manifest.json"


def test_package_proposal_record_rejects_an_invalid_file_digest(tmp_path: Path) -> None:
    proposal, document = _proposal_record(tmp_path)
    manifest_path = proposal / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["prompt.md"] = "not-a-sha256"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    record, issue = trial_bundle_module.package_proposal_record(proposal, document)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/proposal_record/manifest.json/files"


def test_package_proposal_record_rejects_a_failed_proposal(tmp_path: Path) -> None:
    proposal, document = _proposal_record(tmp_path)
    manifest_path = proposal / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "failed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    record, issue = trial_bundle_module.package_proposal_record(proposal, document)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/proposal_record"


def test_package_proposal_record_rejects_a_different_document(tmp_path: Path) -> None:
    proposal, document = _proposal_record(tmp_path)
    document["document_id"] = "different-piece"

    record, issue = trial_bundle_module.package_proposal_record(proposal, document)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path == "/proposal_record/normalized-draft.json"


def test_package_proposal_record_rejects_a_malformed_manifest(tmp_path: Path) -> None:
    proposal, document = _proposal_record(tmp_path)
    (proposal / "manifest.json").write_text("{", encoding="utf-8")

    record, issue = trial_bundle_module.package_proposal_record(proposal, document)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/proposal_record"


def test_package_proposal_record_rejects_a_non_file_entry(tmp_path: Path) -> None:
    proposal, document = _proposal_record(tmp_path)
    prompt_path = proposal / "prompt.md"
    prompt_path.unlink()
    prompt_path.mkdir()

    record, issue = trial_bundle_module.package_proposal_record(proposal, document)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/proposal_record/prompt.md"


def test_package_proposal_record_reports_a_read_failure(tmp_path: Path, monkeypatch) -> None:
    proposal, document = _proposal_record(tmp_path)

    def fail_read_text(path: Path, *args: object, **kwargs: object) -> str:
        raise OSError("read failed")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    record, issue = trial_bundle_module.package_proposal_record(proposal, document)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.STORAGE_ERROR
    assert issue.path == "/proposal_record_dir"


def test_create_trial_bundle_saves_a_self_contained_verified_trial(tmp_path: Path) -> None:
    document = _fixture("approved-script.json")
    response = _fixture("frozen-response.json")
    bundle = tmp_path / "trial-a"

    result = create_trial_bundle(document, response, bundle, trial_id="trial-a")

    assert result.created
    assert result.persisted
    assert result.trial_path == bundle
    assert result.outcome.to_dict() == {
        "terminal_state": "completed",
        "artifact_disposition": "candidate",
        "issues": [],
    }
    assert result.issues == ()
    assert result.bundle_path == bundle
    assert result.manifest_path == bundle / "manifest.json"
    assert result.trial_id == "trial-a"
    assert result.frozen_response_sha256 is not None
    assert json.loads((bundle / "approved-script.json").read_text(encoding="utf-8")) == document
    assert json.loads((bundle / "frozen-response.json").read_text(encoding="utf-8")) == response
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["trial_id"] == "trial-a"
    assert manifest["approved_script"]["document_id"] == document["document_id"]
    assert manifest["approved_script"]["revision"] == document["revision"]
    assert manifest["approved_script"]["content_sha256"] == document["approval"]["content_sha256"]
    assert manifest["realizer"] == {
        "package": "scoim",
        "package_version": "0.1.0",
        "realizer_version": 3,
        "profile": "solo_piano_3m_v1",
    }
    expected_paths = {
        "approved_script": "approved-script.json",
        "frozen_response": "frozen-response.json",
        "musicxml": "artifacts/score.musicxml",
        "smf": "artifacts/final.mid",
        "diagnostics": "artifacts/realization-diagnostics.json",
        "terminal": "terminal.json",
    }
    for name, relative_path in expected_paths.items():
        record = manifest["files"][name]
        assert record["path"] == relative_path
        assert record["sha256"] == _sha256(bundle / relative_path)
    assert json.loads((bundle / "terminal.json").read_text(encoding="utf-8")) == (
        result.outcome.to_dict()
    )
    assert {
        name: manifest["files"][name]["sha256"] for name in ("musicxml", "smf", "diagnostics")
    } == {
        "musicxml": "46e2fe9597ea282783b7492fcf72fb3deaaa121633863be889366484fb041b8d",
        "smf": "752e8379b32d9a0908a424d9c20caaca8703d349c40e0f60f6b3d456c5c3a80a",
        "diagnostics": "28568d243d31fa87af9a5b3d76303d7b49ec51584c51b795983c6bbc55038756",
    }


def test_create_trial_bundle_rejects_an_invalid_staged_realization_record(
    tmp_path: Path,
) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    record, issue = package_staged_realization_record(run_dir, frozen_response)
    assert issue is None
    assert record is not None
    value = json.loads(record)
    parent = value["parent_files"]["run-spec.json"]
    parent["content_base64"] = base64.b64encode(b"{}\n").decode("ascii")
    invalid_record = (json.dumps(value) + "\n").encode("utf-8")

    result = create_trial_bundle(
        _approved_script(),
        frozen_response,
        tmp_path / "trial-invalid-record",
        trial_id="trial-invalid-record",
        realization_record=invalid_record,
    )

    assert result.created is False
    assert result.persisted is False
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/realization_record"


def test_replay_trial_bundle_verifies_a_staged_realization_record(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    record, issue = package_staged_realization_record(run_dir, frozen_response)
    assert issue is None
    assert record is not None
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _approved_script(),
        frozen_response,
        bundle,
        trial_id="staged-record",
        realization_record=record,
    )
    assert created.created is True
    valid_replay = replay_trial_bundle(bundle, tmp_path / "valid-replay")
    assert valid_replay.replayed is True
    record_path = bundle / "model-runs" / "realization.json"
    record_value = json.loads(record_path.read_text(encoding="utf-8"))
    record_value["model"] = "changed-model"
    record_path.write_text(json.dumps(record_value), encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["realization_model_run"]["sha256"] = _sha256(record_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    replayed = replay_trial_bundle(bundle, tmp_path / "replay")

    assert replayed.replayed is False
    assert replayed.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert replayed.issues[0].path == "/realization_record"


def test_fixed_revision_2_bundle_reuses_the_legacy_frozen_response(tmp_path: Path) -> None:
    result = create_trial_bundle(
        _fixture("approved-script-revision-2.json"),
        _fixture("frozen-response.json"),
        tmp_path / "trial",
        trial_id="fixed-revision-2",
    )

    assert result.created is True
    assert result.outcome.promotion_eligible is True
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["realizer"]["realizer_version"] == 3
    replay = replay_trial_bundle(tmp_path / "trial", tmp_path / "replay")
    assert replay.replayed is True
    assert replay.artifact_sha256 == {
        name: manifest["files"][name]["sha256"] for name in ("musicxml", "smf", "diagnostics")
    }


def test_nested_revision_2_bundle_replays_both_typed_requirements(tmp_path: Path) -> None:
    document = json.loads(
        (_NESTED_FIXTURE_DIR / "approved-script-revision-2.json").read_text(encoding="utf-8")
    )
    response = json.loads(
        (_NESTED_FIXTURE_DIR / "frozen-response-revision-2.json").read_text(encoding="utf-8")
    )
    result = create_trial_bundle(
        document,
        response,
        tmp_path / "trial",
        trial_id="nested-revision-2",
    )

    assert result.created is True
    assert result.outcome.promotion_eligible is True
    diagnostics = json.loads(
        (tmp_path / "trial" / "artifacts" / "realization-diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    typed = [
        item
        for item in diagnostics["targets"]
        if item["target_id"].startswith("/script/requirements/")
    ]
    assert len(typed) == 2
    assert {item["status"] for item in typed} == {"passed"}
    replay = replay_trial_bundle(tmp_path / "trial", tmp_path / "replay")
    assert replay.replayed is True
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert replay.artifact_sha256 == {
        name: manifest["files"][name]["sha256"] for name in ("musicxml", "smf", "diagnostics")
    }


def test_two_trial_ids_can_reference_the_same_unchanged_approved_script(
    tmp_path: Path,
) -> None:
    document = _fixture("approved-script.json")
    original = deepcopy(document)
    response = _fixture("frozen-response.json")

    first = create_trial_bundle(document, response, tmp_path / "first", trial_id="first")
    second = create_trial_bundle(document, response, tmp_path / "second", trial_id="second")

    assert first.created and second.created
    assert document == original
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert first_manifest["trial_id"] == "first"
    assert second_manifest["trial_id"] == "second"
    assert first_manifest["approved_script"] == second_manifest["approved_script"]
    assert first_manifest["frozen_response_sha256"] == second_manifest["frozen_response_sha256"]


def test_title_only_edit_keeps_unrelated_score_and_performance_coordinates(
    tmp_path: Path,
) -> None:
    original = _fixture("approved-script.json")
    revised = apply_patch(
        original,
        [
            {"op": "test", "path": "/revision", "value": 1},
            {"op": "replace", "path": "/revision", "value": 2},
            {"op": "add", "path": "/script/requirements", "value": {}},
            {"op": "replace", "path": "/script/title", "value": "別の題名"},
        ],
        new_draft=True,
    )
    assert revised.document is not None
    approved = approve(revised.document)
    assert approved.document is not None
    response = _fixture("frozen-response.json")

    original_script = deepcopy(original["script"])
    revised_script = deepcopy(approved.document["script"])
    assert isinstance(original_script, dict)
    assert isinstance(revised_script, dict)
    original_script.pop("title")
    revised_script.pop("title")
    revised_script.pop("requirements")
    assert revised_script == original_script

    before = create_trial_bundle(original, response, tmp_path / "before", trial_id="before")
    after = create_trial_bundle(
        approved.document,
        response,
        tmp_path / "after",
        trial_id="after",
    )
    assert before.created and after.created

    before_diagnostics = json.loads(
        (tmp_path / "before" / "artifacts" / "realization-diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    after_diagnostics = json.loads(
        (tmp_path / "after" / "artifacts" / "realization-diagnostics.json").read_text(
            encoding="utf-8"
        )
    )

    def artifact_check(diagnostics: dict[str, object], artifact: str, check_id: str) -> object:
        checks = diagnostics["artifacts"][artifact]["checks"]
        return next(check["actual"] for check in checks if check["check_id"] == check_id)

    assert artifact_check(before_diagnostics, "musicxml", "musicxml.notes") == artifact_check(
        after_diagnostics, "musicxml", "musicxml.notes"
    )
    assert artifact_check(before_diagnostics, "smf", "smf.notes") == artifact_check(
        after_diagnostics, "smf", "smf.notes"
    )
    assert artifact_check(before_diagnostics, "smf", "smf.pedals") == artifact_check(
        after_diagnostics, "smf", "smf.pedals"
    )


def test_replay_trial_bundle_reproduces_all_artifacts_without_external_input(
    tmp_path: Path,
) -> None:
    document = _fixture("approved-script.json")
    response = _fixture("frozen-response.json")
    bundle = tmp_path / "trial"
    created = create_trial_bundle(document, response, bundle, trial_id="trial-a")
    assert created.created
    replay = tmp_path / "replay"

    result = replay_trial_bundle(bundle, replay)

    assert result.replayed
    assert result.outcome.to_dict() == created.outcome.to_dict()
    assert result.issues == ()
    assert result.output_path == replay
    assert result.trial_id == "trial-a"
    for filename in ("score.musicxml", "final.mid", "realization-diagnostics.json"):
        assert (replay / filename).read_bytes() == (bundle / "artifacts" / filename).read_bytes()


def test_create_trial_bundle_rejects_an_empty_trial_id(tmp_path: Path) -> None:
    bundle = tmp_path / "trial"

    result = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="  ",
    )

    assert not result.created
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/trial_id"
    assert not bundle.exists()


def test_failed_realization_persists_a_terminal_trial_record(tmp_path: Path) -> None:
    document = _fixture("approved-script.json")
    response = _fixture("frozen-response.json")
    response["schema_version"] = 999
    trial = tmp_path / "failed-trial"

    result = create_trial_bundle(document, response, trial, trial_id="failed-trial")

    assert not result.created
    assert result.persisted
    assert result.trial_path == trial
    assert result.bundle_path is None
    assert result.outcome.terminal_state.value == "failed"
    assert result.outcome.artifact_disposition.value == "none"
    assert json.loads((trial / "terminal.json").read_text(encoding="utf-8")) == (
        result.outcome.to_dict()
    )
    assert (trial / "approved-script.json").is_file()
    assert (trial / "frozen-response.json").is_file()
    assert not (trial / "artifacts").exists()
    manifest = json.loads((trial / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"approved_script", "frozen_response", "terminal"}
    assert manifest["files"]["terminal"]["sha256"] == _sha256(trial / "terminal.json")


def test_schema_invalid_script_persists_a_typed_terminal_record(tmp_path: Path) -> None:
    document = _fixture("approved-script.json")
    document.pop("approval")
    trial = tmp_path / "schema-invalid-trial"

    result = create_trial_bundle(
        document,
        _fixture("frozen-response.json"),
        trial,
        trial_id="schema-invalid-trial",
    )

    assert not result.created
    assert result.persisted
    assert result.issues[0].code is IssueCode.SCHEMA_INVALID
    assert (
        json.loads((trial / "terminal.json").read_text(encoding="utf-8"))["issues"][0]["code"]
        == "schema_invalid"
    )
    assert (trial / "rejected-script.json").is_file()
    assert not (trial / "approved-script.json").exists()


def test_diagnostic_only_trial_is_a_replayable_bundle(tmp_path: Path, monkeypatch) -> None:
    document = _fixture("approved-script.json")
    response = _fixture("frozen-response.json")
    import scoim.realization as realization

    evaluate_quality = realization.evaluate_generic_pipeline_quality
    monkeypatch.setattr(
        "scoim.realization.evaluate_generic_pipeline_quality",
        lambda *args: {**evaluate_quality(*args), "passes": False},
    )
    trial = tmp_path / "diagnostic-trial"

    result = create_trial_bundle(document, response, trial, trial_id="diagnostic-trial")

    assert result.created
    assert result.persisted
    assert result.outcome.artifact_disposition.value == "diagnostic_only"
    assert not result.outcome.promotion_eligible
    assert (trial / "artifacts" / "final.mid").is_file()

    replayed = replay_trial_bundle(trial, tmp_path / "diagnostic-replay")

    assert replayed.replayed
    assert replayed.outcome == result.outcome


def test_create_trial_bundle_does_not_overwrite_an_existing_directory(tmp_path: Path) -> None:
    bundle = tmp_path / "trial"
    bundle.mkdir()
    marker = bundle / "keep.txt"
    marker.write_text("existing\n", encoding="utf-8")

    result = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )

    assert not result.created
    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert marker.read_text(encoding="utf-8") == "existing\n"


def test_create_trial_bundle_reports_a_storage_error_without_partial_output(
    tmp_path: Path,
) -> None:
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied\n", encoding="utf-8")
    bundle = parent_file / "trial"

    result = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )

    assert not result.created
    assert result.issues[0].code is IssueCode.STORAGE_ERROR
    assert not bundle.exists()
    assert parent_file.read_text(encoding="utf-8") == "occupied\n"


def test_replay_trial_bundle_does_not_overwrite_an_existing_directory(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    replay = tmp_path / "replay"
    replay.mkdir()
    marker = replay / "keep.txt"
    marker.write_text("existing\n", encoding="utf-8")

    result = replay_trial_bundle(bundle, replay)

    assert not result.replayed
    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert marker.read_text(encoding="utf-8") == "existing\n"


def test_replay_trial_bundle_rejects_a_new_output_inside_the_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    manifest_before = (bundle / "manifest.json").read_bytes()
    nested_output = bundle / "replayed"

    result = replay_trial_bundle(bundle, nested_output)

    assert not result.replayed
    assert result.issues[0].code is IssueCode.STORAGE_CONFLICT
    assert not nested_output.exists()
    assert (bundle / "manifest.json").read_bytes() == manifest_before


def test_replay_trial_bundle_detects_a_modified_artifact(tmp_path: Path) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    (bundle / "artifacts" / "final.mid").write_bytes(b"changed")
    replay = tmp_path / "replay"

    result = replay_trial_bundle(bundle, replay)

    assert not result.replayed
    assert result.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert result.issues[0].path == "/files/smf/sha256"
    assert not replay.exists()


def test_replay_trial_bundle_detects_a_modified_terminal(tmp_path: Path) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    terminal_path = bundle / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["artifact_disposition"] = "diagnostic_only"
    terminal["issues"] = [
        {
            "code": "projection_target_unmet",
            "message": "tampered",
            "path": "/quality",
        }
    ]
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")

    result = replay_trial_bundle(bundle, tmp_path / "replay")

    assert not result.replayed
    assert result.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert result.issues[0].path == "/files/terminal/sha256"


@pytest.mark.parametrize(
    ("filename", "record_name"),
    (("approved-script.json", "approved_script"), ("frozen-response.json", "frozen_response")),
)
def test_replay_trial_bundle_detects_a_modified_saved_input(
    tmp_path: Path, filename: str, record_name: str
) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    path = bundle / filename
    path.write_bytes(path.read_bytes() + b" ")

    result = replay_trial_bundle(bundle, tmp_path / "replay")

    assert not result.replayed
    assert result.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert result.issues[0].path == f"/files/{record_name}/sha256"


def test_replay_trial_bundle_rejects_malformed_manifest_json(tmp_path: Path) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    (bundle / "manifest.json").write_text("{", encoding="utf-8")
    replay = tmp_path / "replay"

    result = replay_trial_bundle(bundle, replay)

    assert not result.replayed
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/manifest"
    assert not replay.exists()


def test_replay_trial_bundle_rejects_a_non_object_manifest(tmp_path: Path) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    (bundle / "manifest.json").write_text("[]", encoding="utf-8")

    result = replay_trial_bundle(bundle, tmp_path / "replay")

    assert not result.replayed
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/manifest"


def test_replay_trial_bundle_rejects_unknown_manifest_fields(tmp_path: Path) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fallback"] = "accept"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = replay_trial_bundle(bundle, tmp_path / "replay")

    assert not result.replayed
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/manifest/fallback"


def test_replay_trial_bundle_rejects_an_unsupported_realizer_version(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["realizer"]["realizer_version"] = 2
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = replay_trial_bundle(bundle, tmp_path / "replay")

    assert not result.replayed
    assert result.issues[0].code is IssueCode.UNSUPPORTED_PROFILE
    assert result.issues[0].path == "/manifest/realizer"


@pytest.mark.parametrize(
    ("case", "expected_code", "expected_path"),
    (
        ("schema_version", IssueCode.MODEL_OUTPUT_INVALID, "/manifest/bundle_type"),
        ("schema_type", IssueCode.MODEL_OUTPUT_INVALID, "/manifest/schema_version"),
        ("trial_id", IssueCode.MODEL_OUTPUT_INVALID, "/manifest/trial_id"),
        ("approved_fields", IssueCode.MODEL_OUTPUT_INVALID, "/manifest/approved_script"),
        ("approved_values", IssueCode.MODEL_OUTPUT_INVALID, "/manifest/approved_script"),
        ("realizer_fields", IssueCode.MODEL_OUTPUT_INVALID, "/manifest/realizer"),
        ("realizer_type", IssueCode.MODEL_OUTPUT_INVALID, "/manifest/realizer"),
        (
            "response_hash",
            IssueCode.MODEL_OUTPUT_INVALID,
            "/manifest/frozen_response_sha256",
        ),
        ("files", IssueCode.MODEL_OUTPUT_INVALID, "/manifest/files"),
        ("file_record", IssueCode.MODEL_OUTPUT_INVALID, "/manifest/files/smf"),
    ),
)
def test_replay_trial_bundle_rejects_invalid_manifest_values(
    tmp_path: Path, case: str, expected_code: IssueCode, expected_path: str
) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if case == "schema_version":
        manifest["schema_version"] = 4
    elif case == "schema_type":
        manifest["schema_version"] = True
    elif case == "trial_id":
        manifest["trial_id"] = ""
    elif case == "approved_fields":
        manifest["approved_script"] = {}
    elif case == "approved_values":
        manifest["approved_script"]["revision"] = 0
    elif case == "realizer_fields":
        manifest["realizer"] = {}
    elif case == "realizer_type":
        manifest["realizer"]["realizer_version"] = True
    elif case == "response_hash":
        manifest["frozen_response_sha256"] = "invalid"
    elif case == "files":
        manifest["files"] = {}
    elif case == "file_record":
        manifest["files"]["smf"] = {}
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = replay_trial_bundle(bundle, tmp_path / "replay")

    assert not result.replayed
    assert result.issues[0].code is expected_code
    assert result.issues[0].path == expected_path


def test_replay_trial_bundle_rejects_a_manifest_path_outside_the_fixed_layout(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["smf"]["path"] = "../outside.mid"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = replay_trial_bundle(bundle, tmp_path / "replay")

    assert not result.replayed
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/manifest/files/smf"


def test_replay_trial_bundle_detects_an_approved_script_reference_mismatch(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["approved_script"]["content_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = replay_trial_bundle(bundle, tmp_path / "replay")

    assert not result.replayed
    assert result.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert result.issues[0].path == "/manifest/approved_script/content_sha256"


def test_replay_trial_bundle_rejects_a_draft_script_even_when_hashes_are_updated(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    document_path = bundle / "approved-script.json"
    document = json.loads(document_path.read_text(encoding="utf-8"))
    document["status"] = "draft"
    document["approval"] = None
    document["script"]["requirements"] = {}
    document_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["approved_script"]["sha256"] = _sha256(document_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = replay_trial_bundle(bundle, tmp_path / "replay")

    assert not result.replayed
    assert result.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert result.issues[0].path == "/approved_script/status"


def test_replay_trial_bundle_detects_a_frozen_response_content_hash_mismatch(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frozen_response_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = replay_trial_bundle(bundle, tmp_path / "replay")

    assert not result.replayed
    assert result.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert result.issues[0].path == "/frozen_response_sha256"


def test_replay_trial_bundle_rejects_noncanonical_frozen_response_data(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    response_path = bundle / "frozen-response.json"
    response_path.write_text('{"value": NaN}\n', encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["frozen_response"]["sha256"] = _sha256(response_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = replay_trial_bundle(bundle, tmp_path / "replay")

    assert not result.replayed
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/frozen_response"


def test_replay_trial_bundle_rejects_a_required_file_symbolic_link(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path.name == "manifest.json" or original_is_symlink(path),
    )

    result = replay_trial_bundle(bundle, tmp_path / "replay")

    assert not result.replayed
    assert result.issues[0].code is IssueCode.MODEL_OUTPUT_INVALID
    assert result.issues[0].path == "/bundle/manifest.json"


def test_replay_trial_bundle_does_not_publish_a_different_regeneration(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    original_realize = trial_bundle_module.realize_solo_piano_3m

    def changed_realize(*args, **kwargs):
        result = original_realize(*args, **kwargs)
        if result.realized:
            assert result.smf_path is not None
            result.smf_path.write_bytes(result.smf_path.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(trial_bundle_module, "realize_solo_piano_3m", changed_realize)
    replay = tmp_path / "replay"

    result = replay_trial_bundle(bundle, replay)

    assert not result.replayed
    assert result.issues[0].code is IssueCode.LINEAGE_MISMATCH
    assert result.issues[0].path == "/files/smf/sha256"
    assert not replay.exists()


def test_replay_trial_bundle_reports_a_storage_error_without_partial_output(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "trial"
    created = create_trial_bundle(
        _fixture("approved-script.json"),
        _fixture("frozen-response.json"),
        bundle,
        trial_id="trial-a",
    )
    assert created.created
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied\n", encoding="utf-8")
    replay = parent_file / "replay"

    result = replay_trial_bundle(bundle, replay)

    assert not result.replayed
    assert result.issues[0].code is IssueCode.STORAGE_ERROR
    assert not replay.exists()
    assert parent_file.read_text(encoding="utf-8") == "occupied\n"
