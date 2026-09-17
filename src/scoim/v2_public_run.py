"""Stable identity and atomic initialization for one public v2 realization."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_musical_composer.run_state import atomic_write_json

from .runner_identity import RunnerIdentity
from .validation import IssueCode, ValidationIssue


@dataclass(frozen=True, slots=True)
class PublicV2RunRequest:
    """Values that must not change while one public v2 run is resumed."""

    input_kind: str
    input_sha256: str
    composition_id: str
    trial_id: str
    runner_identity: RunnerIdentity


@dataclass(frozen=True, slots=True)
class PublicV2RunResult:
    """Result of initializing or matching one public v2 output root."""

    ready: bool
    resumed: bool
    output_dir: Path | None
    issues: tuple[ValidationIssue, ...]


def ensure_public_v2_run(
    destination: str | Path,
    request: PublicV2RunRequest,
) -> PublicV2RunResult:
    """Atomically create the identity record for a new public v2 run."""
    target = Path(destination).resolve()
    try:
        record = _request_record(request)
    except (TypeError, ValueError) as error:
        return _failure(IssueCode.SCHEMA_INVALID, str(error), target)
    if target.exists():
        marker = target / "public-run.json"
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return _failure(IssueCode.STORAGE_CONFLICT, str(error), marker)
        if saved != record:
            return _failure(
                IssueCode.STORAGE_CONFLICT,
                "The saved public v2 request conflicts with the current request",
                marker,
            )
        return PublicV2RunResult(True, True, target, ())
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent)).resolve()
    try:
        atomic_write_json(temporary / "public-run.json", record)
        os.replace(temporary, target)
        temporary = None
        return PublicV2RunResult(True, False, target, ())
    except OSError as error:
        return _failure(IssueCode.STORAGE_ERROR, str(error), target)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _request_record(request: PublicV2RunRequest) -> dict[str, object]:
    if request.input_kind not in {"flow", "composition"}:
        raise ValueError("unsupported public v2 input kind")
    if not isinstance(request.input_sha256, str) or len(request.input_sha256) != 64:
        raise ValueError("public v2 input hash is invalid")
    if (
        not isinstance(request.composition_id, str)
        or not request.composition_id.strip()
        or not isinstance(request.trial_id, str)
        or not request.trial_id.strip()
    ):
        raise ValueError("public v2 identifiers must not be empty")
    return {
        "schema_version": 1,
        "operation": "scoim-public-v2-realization",
        "input_kind": request.input_kind,
        "input_sha256": request.input_sha256,
        "target_profile": "solo_piano_3m_v2",
        "composition_id": request.composition_id,
        "trial_id": request.trial_id,
        "runner_identity": asdict(request.runner_identity),
    }


def _failure(code: IssueCode, message: str, path: Path) -> PublicV2RunResult:
    return PublicV2RunResult(
        False,
        False,
        None,
        (ValidationIssue(code, message, str(path)),),
    )
