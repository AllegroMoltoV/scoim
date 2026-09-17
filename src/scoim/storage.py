"""Single-file persistence for SCoIM music-script documents."""

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .flow_validation import check_flow
from .validation import CheckResult, IssueCode, ValidationIssue, check


@dataclass(frozen=True, slots=True)
class SaveResult:
    """Result of atomically saving one validated document."""

    saved: bool
    issues: tuple[ValidationIssue, ...]


def _failure(code: IssueCode, message: str, path: str) -> SaveResult:
    return SaveResult(
        saved=False,
        issues=(ValidationIssue(code=code, message=message, path=path),),
    )


def _json_bytes(document: Mapping[str, object]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
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


def save_document(
    path: str | os.PathLike[str],
    document: Mapping[str, object],
    *,
    expected_revision: int | None,
) -> SaveResult:
    """Validate and atomically save one music-script document."""
    return _save_validated_document(path, document, expected_revision, check)


def save_flow_document(
    path: str | os.PathLike[str],
    document: Mapping[str, object],
    *,
    expected_revision: int | None,
) -> SaveResult:
    """Validate and atomically save one human-editable flow document."""
    return _save_validated_document(path, document, expected_revision, check_flow)


def _save_validated_document(
    path: str | os.PathLike[str],
    document: Mapping[str, object],
    expected_revision: int | None,
    checker: Callable[[Mapping[str, object]], CheckResult],
) -> SaveResult:
    validation = checker(document)
    if not validation.valid:
        return SaveResult(saved=False, issues=validation.issues)
    target = Path(path)
    if target.exists():
        try:
            stored_document = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return _failure(IssueCode.STORAGE_ERROR, str(error), "")
        if not isinstance(stored_document, dict):
            return _failure(
                IssueCode.STORAGE_ERROR,
                "The stored document must be a JSON object",
                "",
            )
        if stored_document.get("document_id") != document["document_id"]:
            return _failure(
                IssueCode.STORAGE_CONFLICT,
                "The stored document ID does not match the new document",
                "/document_id",
            )
        if stored_document.get("revision") != expected_revision:
            return _failure(
                IssueCode.STORAGE_CONFLICT,
                "The stored revision does not match the expected revision",
                "/revision",
            )
        if stored_document.get("status") == "approved" and stored_document != document:
            return _failure(
                IssueCode.IMMUTABLE_APPROVED,
                "An approved document cannot be overwritten",
                "/status",
            )
    elif expected_revision is not None:
        return _failure(
            IssueCode.STORAGE_CONFLICT,
            "The document does not exist at the expected revision",
            "/revision",
        )
    try:
        _atomic_write_bytes(target, _json_bytes(document))
    except OSError as error:
        return _failure(IssueCode.STORAGE_ERROR, str(error), "")
    return SaveResult(saved=True, issues=())
