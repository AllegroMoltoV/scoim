import base64
import copy
import hashlib
import json
from pathlib import Path

from test_scoim_staged_realization import QueueRunner, _approved_script, _complete_responses

from scoim.realization_record import (
    package_failed_staged_realization_record,
    package_staged_realization_record,
    verify_failed_staged_realization_record,
    verify_staged_realization_record,
)
from scoim.realization_run import approve_pending_review
from scoim.realization_workspace import workspace_to_dict
from scoim.staged_realization import (
    advance_staged_realization,
    initialize_staged_realization,
)
from scoim.validation import IssueCode, ValidationIssue


def _completed_run(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    run_dir = tmp_path / "staged"
    document = _approved_script()
    initialize_staged_realization(
        run_dir,
        document,
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={},
    )
    status = advance_staged_realization(run_dir, QueueRunner(_complete_responses()))
    assert status.workspace is not None
    frozen_response = {
        "schema_version": 3,
        "profile": "solo_piano_3m_v1",
        "workspace": workspace_to_dict(status.workspace),
    }
    return run_dir, frozen_response


def _completed_repaired_run(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    run_dir = tmp_path / "staged-repaired"
    document = _approved_script()
    responses = _complete_responses()
    valid_accompaniment = responses[4]
    responses[4:5] = [{"accompaniments": []}, valid_accompaniment]
    initialize_staged_realization(
        run_dir,
        document,
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={},
    )
    status = advance_staged_realization(run_dir, QueueRunner(responses))
    assert status.status == "completed"
    assert status.workspace is not None
    return run_dir, {
        "schema_version": 3,
        "profile": "solo_piano_3m_v1",
        "workspace": workspace_to_dict(status.workspace),
    }


def _failed_repaired_run(tmp_path: Path) -> tuple[Path, tuple[ValidationIssue, ...]]:
    run_dir = tmp_path / "staged-failed-repair"
    document = _approved_script()
    responses = _complete_responses()
    responses[4:5] = [{"accompaniments": []}, {"accompaniments": []}]
    initialize_staged_realization(
        run_dir,
        document,
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={},
    )
    status = advance_staged_realization(run_dir, QueueRunner(responses))
    assert status.status == "failed"
    assert status.issues
    return run_dir, status.issues


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _packaged_record_value(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    run_dir, frozen_response = _completed_run(tmp_path)
    record, issue = package_staged_realization_record(run_dir, frozen_response)
    assert issue is None
    assert record is not None
    return json.loads(record), frozen_response


def _completed_reviewed_run(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    run_dir = tmp_path / "staged-reviewed"
    document = _approved_script()
    initialize_staged_realization(
        run_dir,
        document,
        profile="solo_piano_3m_v1",
        model="test-model",
        review_after={"plan-harmony": frozenset({"overall-plan"})},
    )
    runner = QueueRunner(_complete_responses())
    waiting = advance_staged_realization(run_dir, runner)
    assert waiting.status == "awaiting_review"
    approve_pending_review(run_dir / "stages" / "01-plan-harmony", actor="human")
    status = advance_staged_realization(run_dir, runner)
    assert status.workspace is not None
    return run_dir, {
        "schema_version": 3,
        "profile": "solo_piano_3m_v1",
        "workspace": workspace_to_dict(status.workspace),
    }


def test_package_staged_realization_record_collects_a_completed_run(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert issue is None
    assert record is not None
    value = json.loads(record)
    assert value["schema_version"] == 2
    assert value["profile"] == "solo_piano_3m_v1"
    assert [stage["stage_id"] for stage in value["stages"]] == [
        "plan-harmony",
        "melody",
        "accompaniment",
        "ending",
        "performance",
    ]
    assert (
        value["final_workspace_record_sha256"]
        == frozen_response["workspace"]["workspace_record_sha256"]
    )
    packaged_paths = {path for stage in value["stages"] for path in stage["files"]}
    assert ".run.lock" not in packaged_paths


def test_package_staged_realization_record_audits_all_content_repair_attempts(
    tmp_path: Path,
) -> None:
    run_dir, frozen_response = _completed_repaired_run(tmp_path)

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert issue is None
    assert record is not None
    value = json.loads(record)
    accompaniment = next(stage for stage in value["stages"] if stage["stage_id"] == "accompaniment")
    packaged_paths = set(accompaniment["files"])
    assert "attempts/accompaniment-all/attempt-001/validation.json" in packaged_paths
    assert "attempts/accompaniment-all/attempt-002/validation.json" in packaged_paths
    assert verify_staged_realization_record(record, _approved_script(), frozen_response) is None


def test_package_staged_realization_record_rebuilds_the_content_repair_prompt(
    tmp_path: Path,
) -> None:
    run_dir, frozen_response = _completed_repaired_run(tmp_path)
    attempt = (
        run_dir / "stages" / "03-accompaniment" / "attempts" / "accompaniment-all" / "attempt-002"
    )
    changed_prompt = "changed repair prompt"
    (attempt / "prompt.md").write_text(changed_prompt, encoding="utf-8")
    request_path = attempt / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["prompt_sha256"] = hashlib.sha256(changed_prompt.encode("utf-8")).hexdigest()
    request_path.write_text(json.dumps(request), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("/attempt-002/prompt.md")


def test_package_failed_staged_realization_record_rejects_an_invalid_repair_limit(
    tmp_path: Path,
) -> None:
    run_dir, issues = _failed_repaired_run(tmp_path)
    spec_path = run_dir / "stages" / "03-accompaniment" / "run-spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["content_repair_limit"] = "one"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    record, issue = package_failed_staged_realization_record(run_dir, issues)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.STORAGE_ERROR
    assert issue.path == "/realization_record_dir"


def test_package_failed_staged_realization_record_requires_an_operation_list(
    tmp_path: Path,
) -> None:
    run_dir, issues = _failed_repaired_run(tmp_path)
    spec_path = run_dir / "stages" / "03-accompaniment" / "run-spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["operations"] = None
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    record, issue = package_failed_staged_realization_record(run_dir, issues)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.STORAGE_ERROR
    assert issue.path == "/realization_record_dir"


def test_package_staged_realization_record_rejects_a_skipped_repair_attempt(
    tmp_path: Path,
) -> None:
    run_dir, frozen_response = _completed_repaired_run(tmp_path)
    attempt_root = run_dir / "stages" / "03-accompaniment" / "attempts" / "accompaniment-all"
    (attempt_root / "attempt-002").rename(attempt_root / "attempt-003")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.STORAGE_ERROR
    assert issue.path == "/realization_record_dir"


def test_package_staged_realization_record_keeps_one_per_operation_compatibility(
    tmp_path: Path,
) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    parent_spec_path = run_dir / "run-spec.json"
    parent_spec = json.loads(parent_spec_path.read_text(encoding="utf-8"))
    parent_spec["call_policy"] = "one_per_operation"
    parent_spec_path.write_text(json.dumps(parent_spec), encoding="utf-8")
    for stage in parent_spec["stages"]:
        child_dir = run_dir / stage["run_path"]
        child_spec_path = child_dir / "run-spec.json"
        child_spec = json.loads(child_spec_path.read_text(encoding="utf-8"))
        child_spec.pop("content_repair_limit")
        operations = child_spec["operations"]
        child_spec["max_calls"] = sum(
            operation["operation"] not in {"ending", "performance-defaults"}
            for operation in operations
        )
        child_spec_path.write_text(json.dumps(child_spec), encoding="utf-8")
        for validation in child_dir.glob("attempts/*/attempt-*/validation.json"):
            validation.unlink()

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert issue is None
    assert record is not None
    assert verify_staged_realization_record(record, _approved_script(), frozen_response) is None


def test_package_failed_staged_realization_record_keeps_content_repair_attempts(
    tmp_path: Path,
) -> None:
    run_dir, issues = _failed_repaired_run(tmp_path)

    record, issue = package_failed_staged_realization_record(run_dir, issues)

    assert issue is None
    assert record is not None
    value = json.loads(record)
    assert value["failed_stage"] == "accompaniment"
    files = value["files"]
    assert any(path.endswith("attempt-001/validation.json") for path in files)
    assert any(path.endswith("attempt-002/validation.json") for path in files)
    assert verify_failed_staged_realization_record(record, _approved_script()) is None


def test_package_staged_realization_record_detects_a_changed_prompt(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    prompt_path = (
        run_dir
        / "stages"
        / "01-plan-harmony"
        / "attempts"
        / "overall-plan"
        / "attempt-001"
        / "prompt.md"
    )
    prompt_path.write_text("changed prompt", encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("/overall-plan/prompt.md")


def test_package_staged_realization_record_rebuilds_each_checked_diff(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    diff_path = run_dir / "stages" / "01-plan-harmony" / "events" / "overall-plan" / "diff.json"
    event = json.loads(diff_path.read_text(encoding="utf-8"))
    event["payload"]["patch"][0]["value"]["tonal_center"] = 1
    diff_path.write_text(json.dumps(event), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("/overall-plan/diff.json")


def test_packaged_staged_realization_record_verifies_without_the_source_run(
    tmp_path: Path,
) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    record, package_issue = package_staged_realization_record(run_dir, frozen_response)
    assert package_issue is None
    assert record is not None

    issue = verify_staged_realization_record(record, _approved_script(), frozen_response)

    assert issue is None


def test_packaging_does_not_modify_the_source_run(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    before = _tree_hashes(run_dir)

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert issue is None
    assert record is not None
    assert _tree_hashes(run_dir) == before


def test_packaging_rejects_an_unknown_stage_file(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    unknown = run_dir / "stages" / "02-melody" / "unexpected.json"
    unknown.write_text("{}", encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path.endswith("/unexpected.json")


def test_packaging_rejects_a_missing_run_directory(tmp_path: Path) -> None:
    record, issue = package_staged_realization_record(
        tmp_path / "missing",
        {"schema_version": 3, "workspace": {}},
    )

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.STORAGE_ERROR
    assert issue.path == "/realization_record_dir"


def test_packaging_requires_a_version_three_frozen_workspace(tmp_path: Path) -> None:
    run_dir, _ = _completed_run(tmp_path)

    record, issue = package_staged_realization_record(
        run_dir,
        {"schema_version": 2, "workspace": {}},
    )

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/frozen_response"


def test_packaging_rejects_an_invalid_frozen_workspace_hash(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    frozen_response["workspace"]["workspace_record_sha256"] = "not-a-hash"

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/frozen_response/workspace/workspace_record_sha256"


def test_verification_rejects_unknown_record_fields(tmp_path: Path) -> None:
    value, frozen_response = _packaged_record_value(tmp_path)
    value["unknown"] = True

    issue = verify_staged_realization_record(
        json.dumps(value).encode("utf-8"),
        _approved_script(),
        frozen_response,
    )

    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/realization_record"


def test_verification_rejects_a_non_object_file_collection(tmp_path: Path) -> None:
    value, frozen_response = _packaged_record_value(tmp_path)
    value["parent_files"] = []

    issue = verify_staged_realization_record(
        json.dumps(value).encode("utf-8"),
        _approved_script(),
        frozen_response,
    )

    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/realization_record"


def test_verification_rejects_an_unsafe_packaged_path(tmp_path: Path) -> None:
    value, frozen_response = _packaged_record_value(tmp_path)
    parent_files = value["parent_files"]
    assert isinstance(parent_files, dict)
    parent_files["../outside.json"] = next(iter(parent_files.values()))

    issue = verify_staged_realization_record(
        json.dumps(value).encode("utf-8"),
        _approved_script(),
        frozen_response,
    )

    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/realization_record"


def test_verification_rejects_a_packaged_file_hash_mismatch(tmp_path: Path) -> None:
    value, frozen_response = _packaged_record_value(tmp_path)
    parent_files = value["parent_files"]
    assert isinstance(parent_files, dict)
    run_spec = parent_files["run-spec.json"]
    assert isinstance(run_spec, dict)
    run_spec["sha256"] = "0" * 64

    issue = verify_staged_realization_record(
        json.dumps(value).encode("utf-8"),
        _approved_script(),
        frozen_response,
    )

    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/realization_record"


def test_verification_rejects_an_invalid_stage_entry(tmp_path: Path) -> None:
    value, frozen_response = _packaged_record_value(tmp_path)
    stages = value["stages"]
    assert isinstance(stages, list)
    stages[0]["unknown"] = True

    issue = verify_staged_realization_record(
        json.dumps(value).encode("utf-8"),
        _approved_script(),
        frozen_response,
    )

    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/realization_record"


def test_verification_rejects_a_different_approved_script(tmp_path: Path) -> None:
    value, frozen_response = _packaged_record_value(tmp_path)
    different = copy.deepcopy(_approved_script())
    different["script"]["title"] = "別の台本"

    issue = verify_staged_realization_record(
        json.dumps(value).encode("utf-8"),
        different,
        frozen_response,
    )

    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("inputs~1approved-script.json")


def test_verification_rejects_record_metadata_not_backed_by_the_run(tmp_path: Path) -> None:
    value, frozen_response = _packaged_record_value(tmp_path)
    value["model"] = "changed-model"

    issue = verify_staged_realization_record(
        json.dumps(value).encode("utf-8"),
        _approved_script(),
        frozen_response,
    )

    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path == "/realization_record"


def test_packaging_rejects_a_missing_parent_file(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    (run_dir / "run-state.json").unlink()

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path.endswith("/run-state.json")


def test_packaging_rejects_inconsistent_parent_metadata(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    spec_path = run_dir / "run-spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["profile"] = "changed-profile"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path == "/realization_record/parent"


def test_packaging_rejects_a_missing_stage_directory(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    stage = run_dir / "stages" / "01-plan-harmony"
    stage.rename(run_dir / "stages" / "01-plan-harmony-away")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path.endswith("/plan-harmony")


def test_packaging_rejects_a_stage_with_a_different_call_policy(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    spec_path = run_dir / "stages" / "01-plan-harmony" / "run-spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["max_calls"] = 99
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("/plan-harmony/run-spec.json")


def test_packaging_rejects_a_different_frozen_workspace(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    frozen_response["workspace"] = copy.deepcopy(frozen_response["workspace"])
    frozen_response["workspace"]["workspace_record_sha256"] = "0" * 64

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("workspace_record_sha256")


def test_packaging_rejects_a_response_event_that_differs_from_raw_output(
    tmp_path: Path,
) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    response_path = (
        run_dir / "stages" / "01-plan-harmony" / "events" / "overall-plan" / "response.json"
    )
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["payload"]["tonal_center"] = 11
    response_path.write_text(json.dumps(response), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("/overall-plan/response.staged.json")


def test_packaging_rejects_a_terminal_with_a_different_response_hash(
    tmp_path: Path,
) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    terminal_path = (
        run_dir
        / "stages"
        / "01-plan-harmony"
        / "attempts"
        / "overall-plan"
        / "attempt-001"
        / "terminal.json"
    )
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["response_sha256"] = "0" * 64
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("/overall-plan/terminal.json")


def test_packaging_rejects_a_runner_with_a_different_model(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    runner_path = (
        run_dir
        / "stages"
        / "01-plan-harmony"
        / "attempts"
        / "overall-plan"
        / "attempt-001"
        / "runner.json"
    )
    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    runner["model"] = "different-model"
    runner_path.write_text(json.dumps(runner), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("/overall-plan/runner.json")


def test_packaging_rejects_a_response_outside_its_saved_schema(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    operation_root = run_dir / "stages" / "01-plan-harmony"
    attempt_root = operation_root / "attempts" / "overall-plan" / "attempt-001"
    raw_path = attempt_root / "response.staged.json"
    response = json.loads(raw_path.read_text(encoding="utf-8"))
    del response["tonal_center"]
    raw = json.dumps(response).encode("utf-8")
    raw_path.write_bytes(raw)
    event_path = operation_root / "events" / "overall-plan" / "response.json"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["payload"] = response
    event_path.write_text(json.dumps(event), encoding="utf-8")
    terminal_path = attempt_root / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["response_sha256"] = hashlib.sha256(raw).hexdigest()
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path.endswith("/overall-plan/response.staged.json")


def test_packaging_rejects_a_stage_state_that_cannot_be_rebuilt(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    state_path = run_dir / "stages" / "01-plan-harmony" / "realization-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "running"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("/plan-harmony/realization-state.json")


def test_packaging_accepts_a_recorded_human_approval(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_reviewed_run(tmp_path)

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert issue is None
    assert record is not None


def test_packaging_rejects_a_review_for_a_different_diff(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_reviewed_run(tmp_path)
    review_path = run_dir / "stages" / "01-plan-harmony" / "events" / "overall-plan" / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["payload"]["target_diff_sha256"] = "0" * 64
    review_path.write_text(json.dumps(review), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("/overall-plan/review.json")


def test_packaging_rejects_a_replacement_review_without_a_diff(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_reviewed_run(tmp_path)
    review_path = run_dir / "stages" / "01-plan-harmony" / "events" / "overall-plan" / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["payload"]["decision"] = "replace"
    review["payload"]["replacement_diff"] = None
    review_path.write_text(json.dumps(review), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("/overall-plan/review.json")


def test_packaging_rejects_an_approval_with_a_replacement_diff(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_reviewed_run(tmp_path)
    event_root = run_dir / "stages" / "01-plan-harmony" / "events" / "overall-plan"
    review_path = event_root / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    diff = json.loads((event_root / "diff.json").read_text(encoding="utf-8"))
    review["payload"]["replacement_diff"] = diff["payload"]
    review_path.write_text(json.dumps(review), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("/overall-plan/review.json")


def test_verification_rejects_a_non_array_stage_collection(tmp_path: Path) -> None:
    value, frozen_response = _packaged_record_value(tmp_path)
    value["stages"] = {}

    issue = verify_staged_realization_record(
        json.dumps(value).encode("utf-8"),
        _approved_script(),
        frozen_response,
    )

    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/realization_record"


def test_verification_rejects_a_packaged_file_with_missing_fields(tmp_path: Path) -> None:
    value, frozen_response = _packaged_record_value(tmp_path)
    parent_files = value["parent_files"]
    assert isinstance(parent_files, dict)
    parent_files["run-spec.json"] = {"sha256": "0" * 64}

    issue = verify_staged_realization_record(
        json.dumps(value).encode("utf-8"),
        _approved_script(),
        frozen_response,
    )

    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/realization_record"


def test_verification_rejects_a_packaged_file_with_an_invalid_digest(tmp_path: Path) -> None:
    value, frozen_response = _packaged_record_value(tmp_path)
    parent_files = value["parent_files"]
    assert isinstance(parent_files, dict)
    run_spec = parent_files["run-spec.json"]
    assert isinstance(run_spec, dict)
    run_spec["sha256"] = "not-a-hash"

    issue = verify_staged_realization_record(
        json.dumps(value).encode("utf-8"),
        _approved_script(),
        frozen_response,
    )

    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/realization_record"


def test_verification_rejects_a_packaged_file_with_invalid_base64(tmp_path: Path) -> None:
    value, frozen_response = _packaged_record_value(tmp_path)
    parent_files = value["parent_files"]
    assert isinstance(parent_files, dict)
    run_spec = parent_files["run-spec.json"]
    assert isinstance(run_spec, dict)
    run_spec["content_base64"] = "!"

    issue = verify_staged_realization_record(
        json.dumps(value).encode("utf-8"),
        _approved_script(),
        frozen_response,
    )

    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/realization_record"


def test_verification_rejects_a_non_object_json_record() -> None:
    issue = verify_staged_realization_record(b"[]", _approved_script(), {})

    assert issue is not None
    assert issue.code is IssueCode.MODEL_OUTPUT_INVALID
    assert issue.path == "/realization_record"


def test_packaging_reports_malformed_saved_json_as_a_storage_error(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    (run_dir / "run-spec.json").write_text("{", encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.STORAGE_ERROR
    assert issue.path == "/realization_record_dir"


def test_packaging_accepts_a_recorded_replacement_diff(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_reviewed_run(tmp_path)
    event_root = run_dir / "stages" / "01-plan-harmony" / "events" / "overall-plan"
    review_path = event_root / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    diff = json.loads((event_root / "diff.json").read_text(encoding="utf-8"))
    review["payload"]["decision"] = "replace"
    review["payload"]["replacement_diff"] = diff["payload"]
    review_path.write_text(json.dumps(review), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert issue is None
    assert record is not None


def test_packaging_rejects_a_replacement_diff_for_another_workspace(
    tmp_path: Path,
) -> None:
    run_dir, frozen_response = _completed_reviewed_run(tmp_path)
    event_root = run_dir / "stages" / "01-plan-harmony" / "events" / "overall-plan"
    review_path = event_root / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    diff = json.loads((event_root / "diff.json").read_text(encoding="utf-8"))["payload"]
    diff["base_workspace_record_sha256"] = "0" * 64
    review["payload"]["decision"] = "replace"
    review["payload"]["replacement_diff"] = diff
    review_path.write_text(json.dumps(review), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("/overall-plan/diff.json")


def test_packaging_rejects_a_saved_workspace_that_differs_from_events(
    tmp_path: Path,
) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    state_path = run_dir / "stages" / "01-plan-harmony" / "realization-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["workspace"]["unexpected"] = True
    state_path.write_text(json.dumps(state), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path.endswith("/plan-harmony/realization-state.json")


def test_packaging_rejects_an_event_with_the_wrong_type(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    event_path = run_dir / "stages" / "01-plan-harmony" / "events" / "overall-plan" / "request.json"
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["event_type"] = "wrong_type"
    event_path.write_text(json.dumps(event), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.STORAGE_ERROR
    assert issue.path == "/realization_record_dir"


def test_packaging_rejects_an_unknown_saved_operation(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    spec_path = run_dir / "stages" / "01-plan-harmony" / "run-spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["operations"][0]["operation"] = "unknown-operation"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert record is None
    assert issue is not None
    assert issue.code is IssueCode.STORAGE_ERROR
    assert issue.path == "/realization_record_dir"


def test_packaging_does_not_trust_the_derived_child_run_state(tmp_path: Path) -> None:
    run_dir, frozen_response = _completed_run(tmp_path)
    state_path = run_dir / "stages" / "01-plan-harmony" / "run-state.json"
    state_path.write_text("{", encoding="utf-8")

    record, issue = package_staged_realization_record(run_dir, frozen_response)

    assert issue is None
    assert record is not None


def test_verification_propagates_a_reconstructed_run_failure(tmp_path: Path) -> None:
    value, frozen_response = _packaged_record_value(tmp_path)
    parent_files = value["parent_files"]
    assert isinstance(parent_files, dict)
    run_spec = parent_files["run-spec.json"]
    assert isinstance(run_spec, dict)
    decoded = base64.b64decode(run_spec["content_base64"])
    spec = json.loads(decoded)
    spec["profile"] = "changed-profile"
    changed = json.dumps(spec).encode("utf-8")
    run_spec["content_base64"] = base64.b64encode(changed).decode("ascii")
    run_spec["sha256"] = hashlib.sha256(changed).hexdigest()

    issue = verify_staged_realization_record(
        json.dumps(value).encode("utf-8"),
        _approved_script(),
        frozen_response,
    )

    assert issue is not None
    assert issue.code is IssueCode.LINEAGE_MISMATCH
    assert issue.path == "/realization_record/parent"
