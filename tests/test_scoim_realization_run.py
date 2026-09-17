import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from llm_musical_composer.run_state import (
    InterruptedAttemptError,
    RunStore,
    StateConflictError,
)
from scoim.proposal import ProposalRun, RunnerPreflightError
from scoim.realization_run import (
    OperationInstance,
    RealizationRunStore,
    approve_pending_review,
    checked_diff_sha256,
)
from scoim.realization_workspace import (
    CheckedWorkspaceDiff,
    PatchOperation,
    apply_checked_diff,
    checked_diff_to_dict,
    create_workspace,
    state_key_sha256,
    workspace_from_dict,
)
from scoim.validation import IssueCode, ValidationIssue

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "scoim"


def _approved_script() -> dict[str, object]:
    return json.loads(
        (FIXTURE_ROOT / "nested-aba" / "approved-script.json").read_text(encoding="utf-8")
    )


def test_run_can_start_from_a_verified_workspace_snapshot(tmp_path: Path) -> None:
    document = _approved_script()
    initial = create_workspace(document)
    plan_diff = CheckedWorkspaceDiff(
        initial.revision,
        initial.workspace_record_sha256,
        (("/plan", state_key_sha256(initial, "/plan")),),
        ("/plan",),
        (
            PatchOperation(
                "replace",
                "/plan",
                {"tonal_center": 0, "mode": "major", "contrasts": {}},
            ),
        ),
        "0" * 64,
        (),
    )
    applied = apply_checked_diff(initial, plan_diff)
    assert applied.workspace is not None
    starting = applied.workspace
    run = RealizationRunStore(tmp_path / "run")

    run.initialize(
        document,
        operations=(OperationInstance("melody-all", "melody", ("theme",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=1,
        initial_workspace=starting,
        source_workspace_record_sha256="f" * 64,
    )

    status = run.rebuild()
    spec = json.loads((tmp_path / "run" / "run-spec.json").read_text(encoding="utf-8"))
    assert status.workspace == starting
    assert spec["initial_workspace_record_sha256"] == starting.workspace_record_sha256
    assert spec["source_workspace_record_sha256"] == "f" * 64


def test_legacy_v1_run_replays_with_its_original_diff_and_review_hash(tmp_path: Path) -> None:
    document = json.loads(
        (FIXTURE_ROOT / "fixed-aba" / "approved-script.json").read_text(encoding="utf-8")
    )
    legacy_record = {
        "schema_version": 1,
        "approved_script_sha256": (
            "d720cc45cfb022b48fd273cac2b3e2d5fead585a13c5b940a49350cd289b0de4"
        ),
        "revision": 0,
        "plan": None,
        "harmonies": {},
        "provenance": {},
        "music_content_sha256": (
            "a6589502c7924a5e90dd353f59f5d6272c1742ac56bd783f7ba20d68845acdca"
        ),
        "workspace_record_sha256": (
            "bad8dcfb1b63df05f52e7c76526ec79373af42ac95d1d080ad803aa0658adafc"
        ),
    }
    initial = workspace_from_dict(legacy_record)
    run_dir = tmp_path / "legacy-run"
    store = RunStore(run_dir, max_calls=1)
    store.initialize(
        {
            "schema_version": 1,
            "approved_script_sha256": initial.approved_script_sha256,
            "operations": [
                {"instance_id": "overall-plan", "operation": "plan", "targets": ["/plan"]}
            ],
            "review_after": ["overall-plan"],
            "model": "test-model",
            "max_calls": 1,
        }
    )
    store.snapshot_json("inputs/approved-script.json", document)
    store.snapshot_json("inputs/initial-workspace.json", legacy_record)
    checked_diff = CheckedWorkspaceDiff(
        0,
        initial.workspace_record_sha256,
        (("/plan", state_key_sha256(initial, "/plan")),),
        ("/plan",),
        (
            PatchOperation(
                "replace",
                "/plan",
                {"tonal_center": 0, "mode": "major", "contrasts": {}},
            ),
        ),
        "a" * 64,
        (),
    )
    run = RealizationRunStore(run_dir)
    run.record_checked_diff("overall-plan", checked_diff)
    target_hash = checked_diff_sha256(checked_diff)

    run.record_review(
        "overall-plan",
        target_diff_sha256=target_hash,
        decision="approve",
        actor="human",
    )
    completed = run.rebuild()

    saved_diff = json.loads(
        (run_dir / "events" / "overall-plan" / "diff.json").read_text(encoding="utf-8")
    )["payload"]
    assert "write_read_hashes" not in saved_diff
    assert checked_diff_sha256(checked_diff) == target_hash
    assert completed.status == "completed"
    assert completed.workspace.schema_version == 1


def test_run_persists_model_records_and_waits_for_a_requested_review(
    tmp_path: Path,
) -> None:
    document = _approved_script()
    initial = create_workspace(document)
    checked_diff = CheckedWorkspaceDiff(
        base_revision=0,
        base_workspace_record_sha256=initial.workspace_record_sha256,
        read_hashes=(("/plan", state_key_sha256(initial, "/plan")),),
        write_keys=("/plan",),
        patch=(
            PatchOperation(
                "replace",
                "/plan",
                {"tonal_center": 0, "mode": "major", "contrasts": {}},
            ),
        ),
        proposal_sha256="a" * 64,
        validation_issues=(),
    )
    run = RealizationRunStore(tmp_path / "run")
    run.initialize(
        document,
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset({"overall-plan"}),
        model="test-model",
        max_calls=2,
    )

    run.record_request("overall-plan", {"instruction": "choose a plan"})
    run.record_response("overall-plan", {"tonal_center": 0, "mode": "major"})
    run.record_checked_diff("overall-plan", checked_diff)
    status = run.rebuild()

    assert status.status == "awaiting_review"
    assert status.awaiting_review == "overall-plan"
    assert status.workspace == initial
    assert status.completed_operations == ()
    assert (tmp_path / "run" / "events" / "overall-plan" / "request.json").is_file()
    assert (tmp_path / "run" / "events" / "overall-plan" / "response.json").is_file()
    assert (tmp_path / "run" / "events" / "overall-plan" / "diff.json").is_file()
    event_dir = tmp_path / "run" / "events" / "overall-plan"
    assert json.loads((event_dir / "request.json").read_text())["sequence"] == 0
    assert json.loads((event_dir / "response.json").read_text())["sequence"] == 1
    assert json.loads((event_dir / "diff.json").read_text())["sequence"] == 2


def test_a_separate_process_can_approve_and_resume_using_only_the_run_directory(
    tmp_path: Path,
) -> None:
    document = _approved_script()
    initial = create_workspace(document)
    checked_diff = CheckedWorkspaceDiff(
        base_revision=0,
        base_workspace_record_sha256=initial.workspace_record_sha256,
        read_hashes=(("/plan", state_key_sha256(initial, "/plan")),),
        write_keys=("/plan",),
        patch=(
            PatchOperation(
                "replace",
                "/plan",
                {"tonal_center": 9, "mode": "minor", "contrasts": {}},
            ),
        ),
        proposal_sha256="b" * 64,
        validation_issues=(),
    )
    run_dir = tmp_path / "run"
    run = RealizationRunStore(run_dir)
    run.initialize(
        document,
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset({"overall-plan"}),
        model="test-model",
        max_calls=2,
    )
    run.record_request("overall-plan", {"instruction": "choose a plan"})
    run.record_response("overall-plan", {"tonal_center": 9, "mode": "minor"})
    run.record_checked_diff("overall-plan", checked_diff)
    assert run.rebuild().status == "awaiting_review"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import sys; "
                "from scoim.realization_run import approve_pending_review; "
                "approve_pending_review(Path(sys.argv[1]), actor='human')"
            ),
            str(run_dir),
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    resumed = RealizationRunStore(run_dir).rebuild()
    assert resumed.status == "completed"
    assert resumed.completed_operations == ("overall-plan",)
    assert resumed.workspace.plan is not None
    assert resumed.workspace.plan.mode == "minor"
    assert (run_dir / "events" / "overall-plan" / "review.json").is_file()
    assert len(list((run_dir / "events" / "overall-plan").glob("request.json"))) == 1


def test_an_ambiguous_started_model_call_is_not_retried(tmp_path: Path) -> None:
    class TimeoutRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            self.calls += 1
            return ProposalRun(
                provider="fake",
                model="test-model",
                model_settings={"timeout_seconds": 1},
                started=True,
                terminal_state="timeout",
                raw_response=None,
                events=b'{"type":"thread.started"}\n',
                stderr=b"timed out",
                issues=(),
            )

    document = _approved_script()
    run = RealizationRunStore(tmp_path / "run")
    run.initialize(
        document,
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=1,
    )
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")
    runner = TimeoutRunner()

    with pytest.raises(InterruptedAttemptError):
        run.execute_model_call("overall-plan", "choose a plan", schema_path, runner)
    with pytest.raises(InterruptedAttemptError):
        run.execute_model_call("overall-plan", "choose a plan", schema_path, runner)

    assert runner.calls == 1


def test_model_preflight_failure_does_not_reserve_an_attempt(tmp_path: Path) -> None:
    class UnavailableRunner:
        def preflight(self) -> tuple[ValidationIssue, ...]:
            return (
                ValidationIssue(
                    IssueCode.RUNNER_UNAVAILABLE,
                    "runner is unavailable",
                    "/runner",
                ),
            )

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            raise AssertionError("preflight failure must prevent a model call")

    document = _approved_script()
    run_dir = tmp_path / "run"
    run = RealizationRunStore(run_dir)
    run.initialize(
        document,
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=1,
    )
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")

    with pytest.raises(RunnerPreflightError) as caught:
        run.execute_model_call("overall-plan", "choose a plan", schema_path, UnavailableRunner())

    assert caught.value.issues[0].code is IssueCode.RUNNER_UNAVAILABLE
    assert not (run_dir / "events" / "overall-plan" / "request.json").exists()
    assert not (run_dir / "attempts" / "overall-plan").exists()


def test_rebuild_rejects_a_review_for_a_different_diff(tmp_path: Path) -> None:
    document = _approved_script()
    initial = create_workspace(document)
    checked_diff = CheckedWorkspaceDiff(
        0,
        initial.workspace_record_sha256,
        (("/plan", state_key_sha256(initial, "/plan")),),
        ("/plan",),
        (
            PatchOperation(
                "replace",
                "/plan",
                {"tonal_center": 0, "mode": "major", "contrasts": {}},
            ),
        ),
        "c" * 64,
        (),
    )
    run_dir = tmp_path / "run"
    run = RealizationRunStore(run_dir)
    run.initialize(
        document,
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset({"overall-plan"}),
        model="test-model",
        max_calls=2,
    )
    run.record_request("overall-plan", {"instruction": "choose a plan"})
    run.record_response("overall-plan", {"tonal_center": 0, "mode": "major"})
    run.record_checked_diff("overall-plan", checked_diff)
    review_path = run_dir / "events" / "overall-plan" / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sequence": 3,
                "event_type": "review_decision",
                "operation_instance_id": "overall-plan",
                "payload": {
                    "decision": "approve",
                    "actor": "human",
                    "target_diff_sha256": "0" * 64,
                    "replacement_diff": None,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StateConflictError, match="review target"):
        run.rebuild()


def test_model_call_limit_stops_before_invoking_an_extra_runner(tmp_path: Path) -> None:
    class SuccessRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            self.calls += 1
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                b'{"value":1}',
                b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
                b"",
                (),
            )

    run = RealizationRunStore(tmp_path / "run")
    run.initialize(
        _approved_script(),
        operations=(
            OperationInstance("overall-plan", "plan", ("/plan",)),
            OperationInstance("harmony-all", "harmony", ("theme",)),
        ),
        review_after=frozenset(),
        model="test-model",
        max_calls=1,
    )
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")
    runner = SuccessRunner()

    assert run.execute_model_call("overall-plan", "plan", schema_path, runner) == {"value": 1}
    with pytest.raises(RuntimeError, match="call limit"):
        run.execute_model_call("harmony-all", "harmony", schema_path, runner)

    assert runner.calls == 1


def test_reinitializing_a_run_with_a_different_script_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run = RealizationRunStore(run_dir)
    run.initialize(
        _approved_script(),
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )
    other = json.loads(
        (FIXTURE_ROOT / "fixed-aba" / "approved-script.json").read_text(encoding="utf-8")
    )

    with pytest.raises(StateConflictError, match="run-spec"):
        run.initialize(
            other,
            operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
            review_after=frozenset(),
            model="test-model",
            max_calls=2,
        )


def test_rebuild_rejects_a_modified_approved_script_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run = RealizationRunStore(run_dir)
    run.initialize(
        _approved_script(),
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )
    script_path = run_dir / "inputs" / "approved-script.json"
    modified = json.loads(script_path.read_text(encoding="utf-8"))
    modified["script"]["title"] = "modified"
    script_path.write_text(json.dumps(modified), encoding="utf-8")

    with pytest.raises(StateConflictError, match="approved script"):
        run.rebuild()


def test_run_rejects_invalid_operation_and_review_configuration(tmp_path: Path) -> None:
    document = _approved_script()
    with pytest.raises(ValueError, match="unique safe"):
        RealizationRunStore(tmp_path / "unsafe").initialize(
            document,
            operations=(OperationInstance("Unsafe_ID", "plan", ("/plan",)),),
            review_after=frozenset(),
            model="test-model",
            max_calls=2,
        )
    with pytest.raises(ValueError, match="unknown operation"):
        RealizationRunStore(tmp_path / "unknown-review").initialize(
            document,
            operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
            review_after=frozenset({"missing"}),
            model="test-model",
            max_calls=2,
        )
    with pytest.raises(ValueError, match="content_repair_limit"):
        RealizationRunStore(tmp_path / "negative-repair-limit").initialize(
            document,
            operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
            review_after=frozenset(),
            model="test-model",
            max_calls=2,
            content_repair_limit=-1,
        )


def test_review_api_rejects_wrong_targets_and_inconsistent_decisions(
    tmp_path: Path,
) -> None:
    run, checked_diff = _waiting_plan_run(tmp_path / "run")
    digest = checked_diff_sha256(checked_diff)
    with pytest.raises(ValueError, match="approve or replace"):
        run.record_review(
            "overall-plan",
            target_diff_sha256=digest,
            decision="reject",
            actor="human",
        )
    with pytest.raises(ValueError, match="human or llm"):
        run.record_review(
            "overall-plan",
            target_diff_sha256=digest,
            decision="approve",
            actor="robot",
        )
    with pytest.raises(ValueError, match="does not match"):
        run.record_review(
            "overall-plan",
            target_diff_sha256="0" * 64,
            decision="approve",
            actor="human",
        )
    with pytest.raises(ValueError, match="Only a replacement"):
        run.record_review(
            "overall-plan",
            target_diff_sha256=digest,
            decision="replace",
            actor="human",
        )
    wrong_write = replace(
        checked_diff,
        write_keys=("/harmonies/theme",),
        patch=(
            PatchOperation(
                "add",
                "/harmonies/theme",
                [{"duration_units": 1, "root_pitch_class": 0, "quality": "major"}],
            ),
        ),
    )
    with pytest.raises(StateConflictError, match=r"replacement.*declared targets"):
        run.record_review(
            "overall-plan",
            target_diff_sha256=digest,
            decision="replace",
            actor="human",
            replacement_diff=wrong_write,
        )


@pytest.mark.parametrize(
    ("started", "terminal_state", "raw_response", "error_type"),
    [
        (False, "failed", None, RuntimeError),
        (True, "failed", None, RuntimeError),
        (True, "completed", b"{", ValueError),
        (True, "completed", b"[]", ValueError),
    ],
)
def test_model_call_failures_are_persisted_without_a_second_attempt(
    tmp_path: Path,
    started: bool,
    terminal_state: str,
    raw_response: bytes | None,
    error_type: type[Exception],
) -> None:
    class ResultRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            self.calls += 1
            return ProposalRun(
                "fake",
                "test-model",
                {},
                started,
                terminal_state,
                raw_response,
                b'{"type":"thread.started"}\n' if started else b"",
                b"failure",
                (),
            )

    run = RealizationRunStore(tmp_path / "run")
    run.initialize(
        _approved_script(),
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")
    runner = ResultRunner()

    with pytest.raises(error_type):
        run.execute_model_call("overall-plan", "plan", schema_path, runner)
    with pytest.raises(InterruptedAttemptError):
        run.execute_model_call("overall-plan", "plan", schema_path, runner)

    assert runner.calls == 1


def test_model_call_rejects_a_runner_using_a_different_model(tmp_path: Path) -> None:
    class WrongModelRunner:
        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            return ProposalRun(
                "fake",
                "other-model",
                {},
                True,
                "completed",
                b'{"value":1}',
                b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
                b"",
                (),
            )

    run = RealizationRunStore(tmp_path / "run")
    run.initialize(
        _approved_script(),
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")

    with pytest.raises(StateConflictError, match="different model"):
        run.execute_model_call("overall-plan", "plan", schema_path, WrongModelRunner())


def test_completed_model_response_is_reused_without_calling_the_runner_again(
    tmp_path: Path,
) -> None:
    class SuccessRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            self.calls += 1
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                b'{"value":1}',
                b'{"type":"thread.started"}\n{"type":"turn.completed"}\n',
                b"",
                (),
            )

    run = RealizationRunStore(tmp_path / "run")
    run.initialize(
        _approved_script(),
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")
    runner = SuccessRunner()

    first = run.execute_model_call("overall-plan", "plan", schema_path, runner)
    second = run.execute_model_call("overall-plan", "plan", schema_path, runner)
    (tmp_path / "run" / "events" / "overall-plan" / "response.json").unlink()
    recovered = run.execute_model_call("overall-plan", "plan", schema_path, runner)

    assert first == second == recovered == {"value": 1}
    assert runner.calls == 1
    request = json.loads(
        next((tmp_path / "run" / "attempts" / "overall-plan").glob("*/request.json")).read_text(
            encoding="utf-8"
        )
    )
    assert request["requested_model"] == "test-model"


def test_completed_model_call_rejects_a_saved_non_object_response(tmp_path: Path) -> None:
    class SuccessRunner:
        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                b'{"value":1}',
                b'{"type":"turn.completed"}\n',
                b"",
                (),
            )

    run = RealizationRunStore(tmp_path / "run")
    run.initialize(
        _approved_script(),
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")
    run.execute_model_call("overall-plan", "plan", schema_path, SuccessRunner())
    (run.run_dir / "events" / "overall-plan" / "response.json").unlink()
    attempt = next((run.run_dir / "attempts" / "overall-plan").glob("attempt-*"))
    raw_response = b"[]"
    (attempt / "response.staged.json").write_bytes(raw_response)
    terminal_path = attempt / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["response_sha256"] = hashlib.sha256(raw_response).hexdigest()
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")

    with pytest.raises(StateConflictError, match="not an object"):
        run.execute_model_call("overall-plan", "plan", schema_path, SuccessRunner())


def test_one_call_policy_rejects_multiple_saved_attempts(tmp_path: Path) -> None:
    class SuccessRunner:
        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            return ProposalRun(
                "fake",
                "test-model",
                {},
                True,
                "completed",
                b'{"value":1}',
                b'{"type":"turn.completed"}\n',
                b"",
                (),
            )

    run = RealizationRunStore(tmp_path / "run")
    run.initialize(
        _approved_script(),
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")
    run.execute_model_call("overall-plan", "plan", schema_path, SuccessRunner())
    (run.run_dir / "events" / "overall-plan" / "response.json").unlink()
    attempt_root = run.run_dir / "attempts" / "overall-plan"
    shutil.copytree(attempt_root / "attempt-001", attempt_root / "attempt-002")

    with pytest.raises(StateConflictError, match="Multiple model attempts"):
        run.execute_model_call("overall-plan", "plan", schema_path, SuccessRunner())


def test_checked_model_call_reuses_its_saved_final_response(tmp_path: Path) -> None:
    class MustNotRun:
        def run(self, prompt: str, response_schema_path: Path) -> ProposalRun:
            raise AssertionError("a saved final response must not call the model again")

    document = _approved_script()
    initial = create_workspace(document)
    expected_diff = CheckedWorkspaceDiff(
        initial.revision,
        initial.workspace_record_sha256,
        (),
        (),
        (),
        "a" * 64,
        (),
    )
    run = RealizationRunStore(tmp_path / "run")
    run.initialize(
        document,
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
        content_repair_limit=1,
    )
    run.record_response("overall-plan", {"value": 1})
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")

    response, checked_diff = run.execute_checked_model_call(
        "overall-plan",
        "plan",
        schema_path,
        MustNotRun(),
        lambda value: expected_diff,
        lambda prompt, value, issues: "repair",
    )

    assert response == {"value": 1}
    assert checked_diff == expected_diff


def test_checked_model_call_rejects_a_corrupted_repair_limit(tmp_path: Path) -> None:
    run = RealizationRunStore(tmp_path / "run")
    run.initialize(
        _approved_script(),
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
        content_repair_limit=1,
    )
    spec_path = run.run_dir / "run-spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["content_repair_limit"] = "one"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")

    with pytest.raises(StateConflictError, match="invalid content repair limit"):
        run.execute_checked_model_call(
            "overall-plan",
            "plan",
            schema_path,
            object(),
            lambda value: pytest.fail("the response must not be inspected"),
            lambda prompt, value, issues: "repair",
        )


def test_checked_model_call_rejects_attempts_beyond_its_repair_limit(tmp_path: Path) -> None:
    run = RealizationRunStore(tmp_path / "run")
    run.initialize(
        _approved_script(),
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=3,
        content_repair_limit=1,
    )
    attempt_root = run.run_dir / "attempts" / "overall-plan"
    for attempt_number in range(1, 4):
        (attempt_root / f"attempt-{attempt_number:03d}").mkdir(parents=True)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")

    with pytest.raises(StateConflictError, match="exceed the content repair limit"):
        run.execute_checked_model_call(
            "overall-plan",
            "plan",
            schema_path,
            object(),
            lambda value: pytest.fail("the response must not be inspected"),
            lambda prompt, value, issues: "repair",
        )


def test_checked_model_call_does_not_repeat_an_incomplete_saved_attempt(tmp_path: Path) -> None:
    run = RealizationRunStore(tmp_path / "run")
    run.initialize(
        _approved_script(),
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
        content_repair_limit=1,
    )
    (run.run_dir / "attempts" / "overall-plan" / "attempt-001").mkdir(parents=True)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type":"object"}', encoding="utf-8")

    with pytest.raises(InterruptedAttemptError, match="cannot be retried automatically"):
        run.execute_checked_model_call(
            "overall-plan",
            "plan",
            schema_path,
            object(),
            lambda value: pytest.fail("the response must not be inspected"),
            lambda prompt, value, issues: "repair",
        )


def test_direct_approval_and_non_pending_review_paths(tmp_path: Path) -> None:
    run, _ = _waiting_plan_run(tmp_path / "waiting")

    completed = approve_pending_review(run.run_dir, actor="human")

    assert completed.status == "completed"
    with pytest.raises(ValueError, match="not awaiting review"):
        run.pending_checked_diff()
    empty = RealizationRunStore(tmp_path / "empty")
    empty.initialize(
        _approved_script(),
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )
    with pytest.raises(ValueError, match="not awaiting review"):
        empty.pending_checked_diff()
    with pytest.raises(ValueError, match="Unknown operation"):
        empty.record_request("missing", {})


def test_run_reports_a_checked_diff_failure_without_changing_workspace(
    tmp_path: Path,
) -> None:
    document = _approved_script()
    initial = create_workspace(document)
    invalid = CheckedWorkspaceDiff(
        0,
        initial.workspace_record_sha256,
        (),
        ("/plan",),
        (PatchOperation("replace", "/plan", None),),
        "d" * 64,
        (
            ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "invalid proposal",
                "/response",
            ),
        ),
    )
    run = RealizationRunStore(tmp_path / "run")
    run.initialize(
        document,
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )
    run.record_request("overall-plan", {})
    run.record_response("overall-plan", {})
    run.record_checked_diff("overall-plan", invalid)

    failed = run.rebuild()

    assert failed.status == "failed"
    assert failed.workspace == initial
    assert failed.issues == invalid.validation_issues


def test_run_rejects_a_diff_outside_the_declared_operation_targets(
    tmp_path: Path,
) -> None:
    document = _approved_script()
    initial = create_workspace(document)
    wrong_write = CheckedWorkspaceDiff(
        0,
        initial.workspace_record_sha256,
        (),
        ("/harmonies/theme",),
        (
            PatchOperation(
                "add",
                "/harmonies/theme",
                [{"duration_units": 1, "root_pitch_class": 0, "quality": "major"}],
            ),
        ),
        "1" * 64,
        (),
    )
    run = RealizationRunStore(tmp_path / "run")
    run.initialize(
        document,
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset(),
        model="test-model",
        max_calls=2,
    )
    run.record_request("overall-plan", {})
    run.record_response("overall-plan", {})

    with pytest.raises(StateConflictError, match="declared targets"):
        run.record_checked_diff("overall-plan", wrong_write)


@pytest.mark.parametrize("decision", ["approve-with-replacement", "replace-without", "other"])
def test_rebuild_rejects_inconsistent_saved_review_decisions(tmp_path: Path, decision: str) -> None:
    run, checked_diff = _waiting_plan_run(tmp_path / decision)
    payload: dict[str, object] = {
        "decision": "approve",
        "actor": "human",
        "target_diff_sha256": checked_diff_sha256(checked_diff),
        "replacement_diff": None,
    }
    if decision == "approve-with-replacement":
        payload["replacement_diff"] = checked_diff_to_dict(checked_diff)
    elif decision == "replace-without":
        payload["decision"] = "replace"
    else:
        payload["decision"] = "other"
    review_path = run.run_dir / "events" / "overall-plan" / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sequence": 3,
                "event_type": "review_decision",
                "operation_instance_id": "overall-plan",
                "payload": payload,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StateConflictError):
        run.rebuild()


def _waiting_plan_run(
    run_dir: Path,
) -> tuple[RealizationRunStore, CheckedWorkspaceDiff]:
    document = _approved_script()
    initial = create_workspace(document)
    checked_diff = CheckedWorkspaceDiff(
        0,
        initial.workspace_record_sha256,
        (("/plan", state_key_sha256(initial, "/plan")),),
        ("/plan",),
        (
            PatchOperation(
                "replace",
                "/plan",
                {"tonal_center": 0, "mode": "major", "contrasts": {}},
            ),
        ),
        "f" * 64,
        (),
    )
    run = RealizationRunStore(run_dir)
    run.initialize(
        document,
        operations=(OperationInstance("overall-plan", "plan", ("/plan",)),),
        review_after=frozenset({"overall-plan"}),
        model="test-model",
        max_calls=2,
    )
    run.record_request("overall-plan", {"instruction": "choose a plan"})
    run.record_response("overall-plan", {"tonal_center": 0, "mode": "major"})
    run.record_checked_diff("overall-plan", checked_diff)
    return run, checked_diff
