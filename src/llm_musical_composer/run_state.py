"""作曲 run の不変記録、確定書き込み、排他制御を扱う。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, BinaryIO


class StateConflictError(RuntimeError):
    """保存済みの不変記録と、新しい要求が矛盾した。"""


class RunInProgressError(RuntimeError):
    """同じ run を別プロセスが実行している。"""


class InterruptedAttemptError(RuntimeError):
    """外部呼び出しの終了状態を一意に判定できない。"""


_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_CALL_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_bytes(path, _json_bytes(value))


def _write_immutable_bytes(path: Path, content: bytes, label: str) -> None:
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise StateConflictError(f"saved {label} conflicts with current input: {path}")
        return
    atomic_write_bytes(path, content)


def _write_immutable_json(path: Path, value: object, label: str) -> None:
    _write_immutable_bytes(path, _json_bytes(value), label)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StateConflictError(f"JSON record must be an object: {path}")
    return value


def read_jsonl_events(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    if not Path(path).is_file():
        return events, errors
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {number}: {error.msg}")
            continue
        if isinstance(event, dict):
            events.append(event)
        else:
            errors.append(f"line {number}: event is not an object")
    return events, errors


def event_summary(path: Path) -> dict[str, Any]:
    events, parse_errors = read_jsonl_events(path)
    event_types = [event.get("type") for event in events]
    usage = dict.fromkeys(_CALL_FIELDS, 0)
    for event in events:
        candidate = event.get("usage") if event.get("type") == "turn.completed" else None
        if not isinstance(candidate, Mapping):
            continue
        for field in _CALL_FIELDS:
            value = candidate.get(field, 0)
            if isinstance(value, int) and not isinstance(value, bool):
                usage[field] += value
    return {
        "thread_started": "thread.started" in event_types,
        "turn_completed": "turn.completed" in event_types,
        "turn_failed": "turn.failed" in event_types,
        "error_event_count": event_types.count("error"),
        "parse_errors": parse_errors,
        "usage": usage,
    }


class RunLock:
    """同じ run ID の同時実行を OS の助言ロックで拒否する。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle: BinaryIO | None = None

    def __enter__(self) -> RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            self._lock(handle)
        except OSError as error:
            handle.close()
            raise RunInProgressError(
                f"run is already in progress: {self.path.parent.name}"
            ) from error
        self._handle = handle
        return self

    @staticmethod
    def _lock(handle: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover - Windows が本プロジェクトの実行環境
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - Windows が本プロジェクトの実行環境
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        assert self._handle is not None
        try:
            self._unlock(self._handle)
        finally:
            self._handle.close()
            self._handle = None


class RunStore:
    """run 内の不変記録と再構築可能な状態要約を管理する。"""

    def __init__(self, run_dir: Path, *, max_calls: int = 7) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.max_calls = max_calls

    def initialize(self, spec: Mapping[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _write_immutable_json(self.run_dir / "run-spec.json", dict(spec), "run-spec")
        self.rebuild_state()

    def read_spec(self) -> dict[str, Any]:
        return _read_json(self.run_dir / "run-spec.json")

    def snapshot_json(self, relative_path: Path | str, value: object) -> Path:
        target = self.run_dir / relative_path
        _write_immutable_json(target, value, "input snapshot")
        return target

    def snapshot_file(self, relative_path: Path | str, source: Path) -> Path:
        target = self.run_dir / relative_path
        _write_immutable_bytes(target, Path(source).read_bytes(), "input snapshot")
        return target

    def attempt_dirs(self, step_id: str | None = None) -> list[Path]:
        root = self.run_dir / "attempts"
        pattern = f"{step_id}/attempt-*" if step_id is not None else "*/attempt-*"
        return sorted(path for path in root.glob(pattern) if path.is_dir())

    @property
    def call_attempt_count(self) -> int:
        return sum((path / "request.json").is_file() for path in self.attempt_dirs())

    def reserve_attempt(self, step_id: str, request: Mapping[str, Any], prompt: str) -> Path:
        if not _SAFE_ID.fullmatch(step_id):
            raise ValueError(f"unsafe step ID: {step_id}")
        if self.call_attempt_count >= self.max_calls:
            raise RuntimeError("Codex call limit exceeded")
        step_root = self.run_dir / "attempts" / step_id
        step_root.mkdir(parents=True, exist_ok=True)
        number = len(self.attempt_dirs(step_id)) + 1
        destination = step_root / f"attempt-{number:03d}"
        temporary = Path(tempfile.mkdtemp(prefix=".attempt-", dir=step_root))
        try:
            _write_immutable_json(temporary / "request.json", dict(request), "request")
            _write_immutable_bytes(temporary / "prompt.md", prompt.encode("utf-8"), "prompt")
            (temporary / "stdout.jsonl").touch()
            (temporary / "stderr.log").touch()
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self.rebuild_state()
        return destination

    def finalize_attempt(
        self,
        attempt_dir: Path,
        status: str,
        *,
        returncode: int | None,
        response_sha256: str | None = None,
        detail: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"completed", "failed", "interrupted"}:
            raise ValueError(f"invalid terminal status: {status}")
        attempt_dir = Path(attempt_dir).resolve()
        if self.run_dir not in attempt_dir.parents:
            raise ValueError("attempt directory is outside run directory")
        if (attempt_dir / "recovery.json").is_file():
            raise StateConflictError("recovery record prevents terminal finalization")
        summary = event_summary(attempt_dir / "stdout.jsonl")
        terminal = {
            "status": status,
            "returncode": returncode,
            "response_sha256": response_sha256,
            "detail": detail,
            "events": summary,
        }
        _write_immutable_json(attempt_dir / "terminal.json", terminal, "terminal")
        self.rebuild_state()
        return terminal

    def record_recovery_failure(self, attempt_dir: Path, *, detail: str) -> dict[str, str]:
        attempt_dir = Path(attempt_dir).resolve()
        if self.run_dir not in attempt_dir.parents:
            raise ValueError("attempt directory is outside run directory")
        if (attempt_dir / "terminal.json").is_file():
            raise StateConflictError("terminal record prevents recovery record")
        recovery = {
            "status": "recovery_required",
            "detail": detail,
        }
        _write_immutable_json(attempt_dir / "recovery.json", recovery, "recovery")
        self.rebuild_state()
        return recovery

    def record_step(
        self,
        step_id: str,
        status: str,
        input_hashes: Mapping[str, str],
        outputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not _SAFE_ID.fullmatch(step_id):
            raise ValueError(f"unsafe step ID: {step_id}")
        if status not in {"completed", "failed", "interrupted", "skipped"}:
            raise ValueError(f"invalid step status: {status}")
        record = {
            "step_id": step_id,
            "status": status,
            "input_hashes": dict(sorted(input_hashes.items())),
            "outputs": dict(outputs),
        }
        _write_immutable_json(self.run_dir / "steps" / f"{step_id}.json", record, "step")
        self.rebuild_state()
        return record

    def record_reuse(self, step_id: str, attempt_dir: Path) -> None:
        reuse_dir = self.run_dir / "reuses" / step_id
        reuse_dir.mkdir(parents=True, exist_ok=True)
        marker = {
            "step_id": step_id,
            "attempt": str(Path(attempt_dir).relative_to(self.run_dir)),
        }
        _write_immutable_json(reuse_dir / f"reuse-{uuid.uuid4().hex}.json", marker, "reuse")
        self.rebuild_state()

    def promote_file(self, source: Path, target: Path, expected_sha256: str) -> Path:
        content = Path(source).read_bytes()
        return self.promote_bytes(content, target, expected_sha256)

    def promote_bytes(self, content: bytes, target: Path, expected_sha256: str) -> Path:
        target = Path(target)
        if sha256_bytes(content) != expected_sha256:
            raise StateConflictError("staged content fingerprint mismatch")
        if target.exists():
            if sha256_file(target) != expected_sha256:
                raise StateConflictError(f"canonical file conflicts with staged file: {target}")
            return target
        atomic_write_bytes(target, content)
        return target

    def rebuild_state(self) -> dict[str, Any]:
        calls: dict[str, int] = {
            "call_attempt_count": 0,
            "confirmed_external_call_count": 0,
            "successful_external_call_count": 0,
            "preflight_failure_count": 0,
            "failed_attempt_count": 0,
            "failed_external_call_count": 0,
            "interrupted_attempt_count": 0,
            "interrupted_external_call_count": 0,
            "recovery_required_attempt_count": 0,
            "unconfirmed_attempt_count": 0,
            "saved_response_count": sum(
                1 for path in (self.run_dir / "reuses").glob("*/*.json") if path.is_file()
            ),
            **dict.fromkeys(_CALL_FIELDS, 0),
        }
        terminal_statuses: list[str] = []
        for attempt in self.attempt_dirs():
            if not (attempt / "request.json").is_file():
                continue
            calls["call_attempt_count"] += 1
            summary = event_summary(attempt / "stdout.jsonl")
            calls["confirmed_external_call_count"] += int(summary["thread_started"])
            for field in _CALL_FIELDS:
                calls[field] += summary["usage"][field]
            recovery_path = attempt / "recovery.json"
            terminal_path = attempt / "terminal.json"
            if recovery_path.is_file():
                if terminal_path.is_file():
                    raise StateConflictError(
                        f"attempt has both recovery and terminal records: {attempt}"
                    )
                calls["recovery_required_attempt_count"] += 1
                terminal_statuses.append("recovery_required")
                continue
            if not terminal_path.is_file():
                if (summary["turn_completed"] or summary["turn_failed"]) and not summary[
                    "parse_errors"
                ]:
                    calls["recovery_required_attempt_count"] += 1
                    terminal_statuses.append("recovery_required")
                continue
            terminal = _read_json(terminal_path)
            status = terminal.get("status")
            if isinstance(status, str):
                terminal_statuses.append(status)
            if status == "completed":
                calls["successful_external_call_count"] += 1
            elif status == "failed":
                calls["failed_attempt_count"] += 1
                if summary["thread_started"]:
                    calls["failed_external_call_count"] += 1
                else:
                    calls["preflight_failure_count"] += 1
            elif status == "interrupted":
                calls["interrupted_attempt_count"] += 1
                if summary["thread_started"]:
                    calls["interrupted_external_call_count"] += 1
                else:
                    calls["unconfirmed_attempt_count"] += 1
        steps = {
            path.stem: _read_json(path)
            for path in sorted((self.run_dir / "steps").glob("*.json"))
            if path.is_file()
        }
        step_statuses = [step.get("status") for step in steps.values()]
        if any(status == "recovery_required" for status in terminal_statuses):
            status = "recovery_required"
        elif any(status == "interrupted" for status in terminal_statuses + step_statuses):
            status = "interrupted"
        elif any(status == "failed" for status in terminal_statuses + step_statuses):
            status = "failed"
        elif steps.get("publish-final", {}).get("status") == "completed":
            status = "completed"
        elif calls["call_attempt_count"] or steps:
            status = "running"
        else:
            status = "initialized"
        state = {
            "schema_version": 1,
            "status": status,
            "calls": calls,
            "steps": steps,
        }
        atomic_write_json(self.run_dir / "run-state.json", state)
        return state

    def iter_attempt_records(self, step_id: str) -> Iterator[tuple[Path, dict[str, Any]]]:
        for attempt in self.attempt_dirs(step_id):
            request_path = attempt / "request.json"
            if request_path.is_file():
                yield attempt, _read_json(request_path)
