from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_musical_composer.run_state import (
    RunInProgressError,
    RunLock,
    RunStore,
    StateConflictError,
    atomic_write_json,
    event_summary,
    sha256_file,
    sha256_json,
)


def test_run_spec_is_immutable_and_state_is_rebuilt_from_records(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run", max_calls=3)
    spec = {"schema_version": 1, "run_id": "run-1", "model": "model-a"}

    store.initialize(spec)
    store.initialize(spec)
    store.record_step("prepare", "completed", {"profile": "abc"}, {"path": "input.json"})
    (store.run_dir / "run-state.json").unlink()

    state = store.rebuild_state()

    assert state["status"] == "running"
    assert state["steps"]["prepare"]["status"] == "completed"
    assert json.loads((store.run_dir / "run-state.json").read_text(encoding="utf-8")) == state
    with pytest.raises(StateConflictError, match="run-spec"):
        store.initialize({**spec, "model": "model-b"})


def test_attempt_records_are_immutable_and_counts_remain_distinct(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run", max_calls=3)
    store.initialize({"schema_version": 1})
    completed = store.reserve_attempt("compose-1", {"prompt_sha256": "one"}, "prompt 1")
    (completed / "stdout.jsonl").write_text(
        json.dumps({"type": "thread.started"})
        + "\n"
        + json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": 2,
                    "output_tokens": 5,
                    "reasoning_output_tokens": 3,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    response = completed / "response.staged.json"
    response.write_text('{"composition_source":"x","intent_summary":"y"}', encoding="utf-8")
    store.finalize_attempt(
        completed,
        "completed",
        returncode=0,
        response_sha256=sha256_file(response),
    )
    interrupted = store.reserve_attempt("compose-2", {"prompt_sha256": "two"}, "prompt 2")
    store.finalize_attempt(interrupted, "interrupted", returncode=None)

    counts = store.rebuild_state()["calls"]

    assert counts == {
        "call_attempt_count": 2,
        "confirmed_external_call_count": 1,
        "successful_external_call_count": 1,
        "preflight_failure_count": 0,
        "failed_attempt_count": 0,
        "failed_external_call_count": 0,
        "interrupted_attempt_count": 1,
        "interrupted_external_call_count": 0,
        "recovery_required_attempt_count": 0,
        "unconfirmed_attempt_count": 1,
        "saved_response_count": 0,
        "input_tokens": 10,
        "cached_input_tokens": 2,
        "output_tokens": 5,
        "reasoning_output_tokens": 3,
    }
    with pytest.raises(StateConflictError, match="terminal"):
        store.finalize_attempt(interrupted, "failed", returncode=1)


def test_recovery_failure_is_immutable_and_blocks_normal_state_rebuild(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.initialize({"run_id": "recovery-test"})
    attempt = store.reserve_attempt("piece-plan", {"model": "test"}, "prompt")

    recovery = store.record_recovery_failure(
        attempt,
        detail="timed out process could not be terminated",
    )

    assert recovery == {
        "status": "recovery_required",
        "detail": "timed out process could not be terminated",
    }
    assert not (attempt / "terminal.json").exists()
    state = store.rebuild_state()
    assert state["status"] == "recovery_required"
    assert state["calls"]["recovery_required_attempt_count"] == 1

    with pytest.raises(StateConflictError, match="recovery"):
        store.record_recovery_failure(attempt, detail="different detail")


def test_completed_event_without_terminal_is_recovery_required(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.initialize({"run_id": "completion-gap"})
    attempt = store.reserve_attempt("piece-plan", {"model": "test"}, "prompt")
    (attempt / "stdout.jsonl").write_text(
        '{"type":"thread.started"}\n{"type":"turn.completed"}\n', encoding="utf-8"
    )

    state = store.rebuild_state()

    assert state["status"] == "recovery_required"
    assert state["calls"]["recovery_required_attempt_count"] == 1


def test_failed_event_without_terminal_is_recovery_required(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.initialize({"run_id": "failure-gap"})
    attempt = store.reserve_attempt("piece-plan", {"model": "test"}, "prompt")
    (attempt / "stdout.jsonl").write_text(
        '{"type":"thread.started"}\n{"type":"turn.failed"}\n', encoding="utf-8"
    )

    state = store.rebuild_state()

    assert state["status"] == "recovery_required"
    assert state["calls"]["recovery_required_attempt_count"] == 1


def test_attempt_limit_uses_reserved_attempts(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run", max_calls=1)
    store.initialize({"schema_version": 1})
    store.reserve_attempt("compose-1", {"prompt_sha256": "one"}, "prompt")

    with pytest.raises(RuntimeError, match="limit"):
        store.reserve_attempt("compose-2", {"prompt_sha256": "two"}, "prompt")


def test_atomic_write_and_promotion_do_not_leave_temporary_files(tmp_path: Path) -> None:
    target = tmp_path / "value.json"
    atomic_write_json(target, {"value": 1})
    atomic_write_json(target, {"value": 2})
    source = tmp_path / "source.bin"
    source.write_bytes(b"music")
    store = RunStore(tmp_path / "run")

    store.promote_file(source, store.run_dir / "final.bin", sha256_file(source))

    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 2}
    assert (store.run_dir / "final.bin").read_bytes() == b"music"
    assert list(tmp_path.glob("*.tmp")) == []
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_run_lock_rejects_second_holder_and_releases(tmp_path: Path) -> None:
    lock_path = tmp_path / "run" / ".run.lock"
    with RunLock(lock_path), pytest.raises(RunInProgressError), RunLock(lock_path):
        pytest.fail("second lock unexpectedly succeeded")

    with RunLock(lock_path):
        assert lock_path.is_file()


def test_run_lock_releases_when_protected_body_raises(tmp_path: Path) -> None:
    lock_path = tmp_path / "run" / ".run.lock"

    with pytest.raises(RuntimeError, match="body failed"), RunLock(lock_path):
        raise RuntimeError("body failed")

    with RunLock(lock_path):
        assert lock_path.is_file()


def test_event_summary_preserves_parse_errors_and_ignores_invalid_usage(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\nnot-json\n[]\n"
        + json.dumps({"type": "error", "usage": {"input_tokens": 999}})
        + "\n"
        + json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": True, "output_tokens": "unknown"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = event_summary(path)

    assert len(summary["parse_errors"]) == 2
    assert summary["error_event_count"] == 1
    assert summary["usage"]["input_tokens"] == 0
    assert event_summary(tmp_path / "missing.jsonl")["turn_completed"] is False


def test_run_store_rejects_invalid_records_and_conflicting_promotions(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    target = store.run_dir / "target.bin"

    store.snapshot_json("inputs/value.json", {"value": 1})
    with pytest.raises(StateConflictError, match="snapshot"):
        store.snapshot_json("inputs/value.json", {"value": 2})
    with pytest.raises(ValueError, match="step ID"):
        store.reserve_attempt("../escape", {}, "prompt")
    with pytest.raises(ValueError, match="terminal status"):
        store.finalize_attempt(tmp_path, "unknown", returncode=None)
    with pytest.raises(ValueError, match="outside"):
        store.finalize_attempt(tmp_path, "failed", returncode=1)
    with pytest.raises(ValueError, match="step ID"):
        store.record_step("bad/id", "completed", {}, {})
    with pytest.raises(ValueError, match="step status"):
        store.record_step("valid-id", "running", {}, {})
    with pytest.raises(StateConflictError, match="fingerprint"):
        store.promote_file(source, target, "0" * 64)

    store.promote_file(source, target, sha256_file(source))
    assert store.promote_file(source, target, sha256_file(source)) == target
    target.write_bytes(b"changed")
    with pytest.raises(StateConflictError, match="conflicts"):
        store.promote_file(source, target, sha256_file(source))


def test_failed_atomic_replacement_removes_temporary_artifacts(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "value.json"
    monkeypatch.setattr(
        "llm_musical_composer.run_state.os.replace",
        lambda *args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_json(target, {"value": 1})

    assert not target.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_failed_attempt_reservation_does_not_consume_call_budget(
    tmp_path: Path, monkeypatch
) -> None:
    store = RunStore(tmp_path / "run")
    store.initialize({"schema_version": 1})
    monkeypatch.setattr(
        "llm_musical_composer.run_state.os.replace",
        lambda *args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        store.reserve_attempt("compose-1", {}, "prompt")

    assert store.call_attempt_count == 0
    assert list((store.run_dir / "attempts" / "compose-1").iterdir()) == []


@pytest.mark.parametrize(
    ("step_status", "expected"),
    [("completed", "completed"), ("failed", "failed"), ("interrupted", "interrupted")],
)
def test_run_state_derives_terminal_run_status(
    tmp_path: Path, step_status: str, expected: str
) -> None:
    store = RunStore(tmp_path / step_status)
    store.initialize({"schema_version": 1})
    store.record_step("publish-final", step_status, {}, {})

    assert store.rebuild_state()["status"] == expected
