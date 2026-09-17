"""Immutable generation-trial bundles and offline replay."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import rfc8785

from .outcome import ArtifactDisposition, ExecutionOutcome, TerminalState
from .realization import realize_solo_piano_3m
from .realization_record import (
    verify_failed_staged_realization_record,
    verify_staged_realization_record,
)
from .realization_script import (
    check_realization_script,
    is_validated_script,
    realization_script_sha256,
)
from .validation import IssueCode, ValidationIssue, check, content_sha256

_PACKAGE_VERSION = "0.1.0"
_CURRENT_REALIZER_VERSION = 4
_REALIZER_VERSION_BY_RESPONSE_SCHEMA = {1: 3, 2: 4, 3: 5}
_RESPONSE_SCHEMA_BY_REALIZER_VERSION = {
    value: key for key, value in _REALIZER_VERSION_BY_RESPONSE_SCHEMA.items()
}
_SUPPORTED_REALIZER_VERSIONS = frozenset(_RESPONSE_SCHEMA_BY_REALIZER_VERSION)
_PROFILE = "solo_piano_3m_v1"
_FILE_PATHS = {
    "approved_script": "approved-script.json",
    "frozen_response": "frozen-response.json",
    "musicxml": "artifacts/score.musicxml",
    "smf": "artifacts/final.mid",
    "diagnostics": "artifacts/realization-diagnostics.json",
    "terminal": "terminal.json",
}
_DIAGNOSTIC_FILE_PATHS = {
    "phase_04_melody": "artifacts/phase-04-melody.mid",
    "phase_06_score": "artifacts/phase-06-score.mid",
}
_VALIDATED_SCRIPT_PATH = "validated-script.json"
_SOURCE_COMPOSITION_MANIFEST_PATH = "source-composition-manifest.json"
_BUNDLE_TYPE = "realization"
_CURRENT_BUNDLE_SCHEMA_VERSION = 4
_MODEL_RUN_PATHS = {
    "realization_model_run": "model-runs/realization.json",
    "proposal_model_run": "model-runs/proposal.json",
}
_PROPOSAL_RECORD_REQUIRED_FILES = frozenset(
    {
        "normalized-draft.json",
        "prompt.md",
        "request.json",
        "response-schema.json",
        "validation.json",
    }
)
_PROPOSAL_RECORD_OPTIONAL_FILES = frozenset(
    {"raw-response.txt", "runner-events.jsonl", "stderr.log"}
)
_REJECTED_SCRIPT_PATH = "rejected-script.json"
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "trial_id",
        "approved_script",
        "realizer",
        "frozen_response_sha256",
        "files",
    }
)
_CURRENT_MANIFEST_FIELDS = frozenset(
    {
        "bundle_type",
        "schema_version",
        "trial_id",
        "validated_script",
        "source_composition",
        "realizer",
        "frozen_response_sha256",
        "files",
    }
)


@dataclass(frozen=True, slots=True)
class TrialBundleResult:
    """Result of creating one immutable generation-trial bundle."""

    created: bool
    persisted: bool
    trial_path: Path | None
    outcome: ExecutionOutcome
    bundle_path: Path | None
    manifest_path: Path | None
    trial_id: str | None
    frozen_response_sha256: str | None

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        """Return typed reasons from the single outcome source."""

        return self.outcome.issues


@dataclass(frozen=True, slots=True)
class TrialReplayResult:
    """Result of replaying a saved generation-trial bundle."""

    replayed: bool
    outcome: ExecutionOutcome
    output_path: Path | None
    trial_id: str | None
    artifact_sha256: Mapping[str, str]

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        """Return typed reasons from the single outcome source."""

        return self.outcome.issues


def _failure(
    code: IssueCode,
    message: str,
    path: str,
    *,
    trial_id: str | None = None,
) -> TrialBundleResult:
    issue = ValidationIssue(code=code, message=message, path=path)
    return TrialBundleResult(
        created=False,
        persisted=False,
        trial_path=None,
        outcome=ExecutionOutcome(
            terminal_state=TerminalState.FAILED,
            artifact_disposition=ArtifactDisposition.NONE,
            issues=(issue,),
        ),
        bundle_path=None,
        manifest_path=None,
        trial_id=trial_id,
        frozen_response_sha256=None,
    )


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_proposal_record(
    proposal_dir: Path,
    document: Mapping[str, object],
) -> tuple[bytes | None, ValidationIssue | None]:
    """Validate and package one completed proposal record for a generation trial."""

    proposal = Path(proposal_dir)
    try:
        if proposal.is_symlink() or not proposal.is_dir():
            return None, ValidationIssue(
                IssueCode.STORAGE_ERROR,
                "The proposal record must be a real directory",
                "/proposal_record_dir",
            )
        entries = {entry.name: entry for entry in proposal.iterdir()}
        manifest_path = entries.get("manifest.json")
        if manifest_path is None or manifest_path.is_symlink() or not manifest_path.is_file():
            return None, ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "The proposal record manifest is missing or linked",
                "/proposal_record/manifest.json",
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "status",
            "runner",
            "draft",
            "files",
        }:
            return None, ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "The proposal record manifest fields are invalid",
                "/proposal_record/manifest.json",
            )
        raw_files = manifest["files"]
        if not isinstance(raw_files, dict) or any(
            not isinstance(name, str) or not _is_sha256(digest)
            for name, digest in raw_files.items()
        ):
            return None, ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "The proposal record file manifest is invalid",
                "/proposal_record/manifest.json/files",
            )
        names = set(raw_files)
        allowed = _PROPOSAL_RECORD_REQUIRED_FILES | _PROPOSAL_RECORD_OPTIONAL_FILES
        if (
            manifest["schema_version"] != 1
            or manifest["status"] != "completed"
            or not _PROPOSAL_RECORD_REQUIRED_FILES.issubset(names)
            or not names.issubset(allowed)
            or set(entries) != names | {"manifest.json"}
        ):
            return None, ValidationIssue(
                IssueCode.MODEL_OUTPUT_INVALID,
                "The proposal record is not a complete supported record",
                "/proposal_record",
            )
        packaged_files: dict[str, dict[str, str]] = {}
        for name in sorted(names | {"manifest.json"}):
            path = entries[name]
            if path.is_symlink() or not path.is_file():
                return None, ValidationIssue(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "A proposal record file is missing or linked",
                    f"/proposal_record/{name}",
                )
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if name != "manifest.json" and digest != raw_files[name]:
                return None, ValidationIssue(
                    IssueCode.LINEAGE_MISMATCH,
                    "A proposal record file hash does not match its manifest",
                    f"/proposal_record/{name}",
                )
            packaged_files[name] = {
                "sha256": digest,
                "content_base64": base64.b64encode(content).decode("ascii"),
            }
        draft = json.loads(entries["normalized-draft.json"].read_text(encoding="utf-8"))
        draft_reference = manifest["draft"]
        draft_validation = check(draft) if isinstance(draft, dict) else None
        if (
            not isinstance(draft, dict)
            or draft_validation is None
            or not draft_validation.valid
            or draft.get("status") != "draft"
            or draft.get("approval") is not None
            or not isinstance(draft_reference, dict)
            or set(draft_reference) != {"document_id", "revision", "content_sha256"}
            or draft.get("document_id") != document.get("document_id")
            or draft_reference.get("document_id") != document.get("document_id")
            or draft_reference.get("revision") != draft.get("revision")
            or draft_reference.get("content_sha256") != content_sha256(draft)
        ):
            return None, ValidationIssue(
                IssueCode.LINEAGE_MISMATCH,
                "The proposal draft does not match the approved script lineage",
                "/proposal_record/normalized-draft.json",
            )
    except json.JSONDecodeError as error:
        return None, ValidationIssue(
            IssueCode.MODEL_OUTPUT_INVALID,
            str(error),
            "/proposal_record",
        )
    except (OSError, UnicodeDecodeError, TypeError, ValueError) as error:
        return None, ValidationIssue(
            IssueCode.STORAGE_ERROR,
            str(error),
            "/proposal_record_dir",
        )
    return _json_bytes({"schema_version": 1, "files": packaged_files}), None


def _replay_failure(
    code: IssueCode,
    message: str,
    path: str,
    *,
    trial_id: str | None = None,
) -> TrialReplayResult:
    issue = ValidationIssue(code=code, message=message, path=path)
    return TrialReplayResult(
        replayed=False,
        outcome=ExecutionOutcome(
            terminal_state=TerminalState.FAILED,
            artifact_disposition=ArtifactDisposition.NONE,
            issues=(issue,),
        ),
        output_path=None,
        trial_id=trial_id,
        artifact_sha256={},
    )


def _load_json_object(path: Path, issue_path: str) -> dict[str, object] | TrialReplayResult:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        return _replay_failure(IssueCode.STORAGE_ERROR, str(error), issue_path)
    except json.JSONDecodeError as error:
        return _replay_failure(IssueCode.MODEL_OUTPUT_INVALID, str(error), issue_path)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        return _replay_failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The saved JSON value must be an object",
            issue_path,
        )
    return value


def _object(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        return None
    return dict(value)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_composition_manifest(
    document: Mapping[str, object],
    composition_manifest: bytes | None,
    *,
    trial_id: str,
) -> tuple[dict[str, object] | None, TrialBundleResult | None]:
    try:
        raw_manifest = json.loads(composition_manifest.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, _failure(
            IssueCode.LINEAGE_MISMATCH,
            f"A validated script trial requires its composition manifest: {error}",
            "/composition_manifest",
            trial_id=trial_id,
        )
    manifest = _object(raw_manifest)
    if (
        manifest is None
        or manifest.get("bundle_type") != "composition"
        or manifest.get("schema_version") != 2
        or manifest.get("terminal_state") != "succeeded"
        or manifest.get("composition_id") != document.get("document_id")
        or manifest.get("validated_script_content_sha256") != realization_script_sha256(document)
    ):
        return None, _failure(
            IssueCode.LINEAGE_MISMATCH,
            "The composition manifest does not identify the validated script",
            "/composition_manifest",
            trial_id=trial_id,
        )
    return manifest, None


def _manifest_issue(
    manifest: Mapping[str, object],
) -> tuple[str | None, TrialReplayResult | None]:
    schema_version = manifest.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        return None, _replay_failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The trial bundle schema version must be an integer",
            "/manifest/schema_version",
        )
    if schema_version not in {3, _CURRENT_BUNDLE_SCHEMA_VERSION}:
        return None, _replay_failure(
            IssueCode.UNSUPPORTED_PROFILE,
            "The trial bundle schema version is not supported",
            "/manifest/schema_version",
        )
    current = schema_version == _CURRENT_BUNDLE_SCHEMA_VERSION
    if current and manifest.get("bundle_type") != _BUNDLE_TYPE:
        return None, _replay_failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The realization bundle type is invalid",
            "/manifest/bundle_type",
        )
    expected_manifest_fields = _CURRENT_MANIFEST_FIELDS if current else _MANIFEST_FIELDS
    extra = set(manifest) - expected_manifest_fields
    missing = expected_manifest_fields - set(manifest)
    if extra or missing:
        field = sorted(extra or missing)[0]
        return None, _replay_failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The manifest fields do not match the bundle contract",
            f"/manifest/{field}",
        )
    trial_id = manifest["trial_id"]
    if not isinstance(trial_id, str) or not trial_id.strip():
        return None, _replay_failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The trial ID must be a non-empty string",
            "/manifest/trial_id",
        )
    script_field = "validated_script" if current else "approved_script"
    approved = _object(manifest[script_field])
    if approved is None or set(approved) != {"document_id", "revision", "content_sha256"}:
        return None, _replay_failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The approved-script reference is invalid",
            f"/manifest/{script_field}",
            trial_id=trial_id,
        )
    if (
        not isinstance(approved["document_id"], str)
        or not approved["document_id"]
        or isinstance(approved["revision"], bool)
        or not isinstance(approved["revision"], int)
        or approved["revision"] < 1
        or not _is_sha256(approved["content_sha256"])
    ):
        return None, _replay_failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The approved-script reference values are invalid",
            f"/manifest/{script_field}",
            trial_id=trial_id,
        )
    if current:
        source_composition = _object(manifest["source_composition"])
        if (
            source_composition is None
            or set(source_composition) != {"manifest_sha256"}
            or not _is_sha256(source_composition["manifest_sha256"])
        ):
            return None, _replay_failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                "The source composition reference is invalid",
                "/manifest/source_composition",
                trial_id=trial_id,
            )
    realizer = _object(manifest["realizer"])
    expected_realizer = {
        "package": "scoim",
        "package_version": _PACKAGE_VERSION,
        "profile": _PROFILE,
    }
    if realizer is None or set(realizer) != set(expected_realizer) | {"realizer_version"}:
        return None, _replay_failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The realizer fields are invalid",
            "/manifest/realizer",
            trial_id=trial_id,
        )
    realizer_types_match = all(
        type(realizer[field]) is type(value) for field, value in expected_realizer.items()
    )
    if not realizer_types_match or (
        isinstance(realizer["realizer_version"], bool)
        or not isinstance(realizer["realizer_version"], int)
    ):
        return None, _replay_failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The realizer value types are invalid",
            "/manifest/realizer",
            trial_id=trial_id,
        )
    if (
        any(realizer[field] != value for field, value in expected_realizer.items())
        or realizer["realizer_version"] not in _SUPPORTED_REALIZER_VERSIONS
    ):
        return None, _replay_failure(
            IssueCode.UNSUPPORTED_PROFILE,
            "The saved realizer is not supported",
            "/manifest/realizer",
            trial_id=trial_id,
        )
    if not _is_sha256(manifest["frozen_response_sha256"]):
        return None, _replay_failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The frozen-response content hash is invalid",
            "/manifest/frozen_response_sha256",
            trial_id=trial_id,
        )
    file_records = _object(manifest["files"])
    expected_paths = {**_FILE_PATHS, **_DIAGNOSTIC_FILE_PATHS, **_MODEL_RUN_PATHS}
    required_file_names = set(_FILE_PATHS)
    if current:
        expected_paths.pop("approved_script")
        expected_paths["validated_script"] = _VALIDATED_SCRIPT_PATH
        expected_paths["source_composition_manifest"] = _SOURCE_COMPOSITION_MANIFEST_PATH
        required_file_names.remove("approved_script")
        required_file_names.add("validated_script")
        required_file_names.add("source_composition_manifest")
        required_file_names.update(_DIAGNOSTIC_FILE_PATHS)
    if (
        file_records is None
        or not required_file_names.issubset(file_records)
        or not set(file_records).issubset(set(expected_paths))
    ):
        return None, _replay_failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The bundle file records are invalid",
            "/manifest/files",
            trial_id=trial_id,
        )
    for name, raw_record in file_records.items():
        record = _object(raw_record)
        if record is None or set(record) != {"path", "sha256"}:
            return None, _replay_failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                "A bundle file record is invalid",
                f"/manifest/files/{name}",
                trial_id=trial_id,
            )
        if record["path"] != expected_paths[name] or not _is_sha256(record["sha256"]):
            return None, _replay_failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                "A bundle file path or hash is invalid",
                f"/manifest/files/{name}",
                trial_id=trial_id,
            )
    return trial_id, None


def _bundle_layout_issue(
    bundle: Path, *, validated_script: bool = False
) -> TrialReplayResult | None:
    bundle_resolved = bundle.resolve()
    artifacts = bundle / "artifacts"
    if artifacts.is_symlink() or not artifacts.is_dir():
        return _replay_failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The bundle artifacts path must be a real directory",
            "/bundle/artifacts",
        )
    required = {"manifest.json": bundle / "manifest.json"}
    file_paths = dict(_FILE_PATHS)
    if validated_script:
        file_paths["approved_script"] = _VALIDATED_SCRIPT_PATH
        file_paths["source_composition_manifest"] = _SOURCE_COMPOSITION_MANIFEST_PATH
    required.update({relative: bundle / relative for relative in file_paths.values()})
    for relative_path, path in required.items():
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(bundle_resolved)
        ):
            return _replay_failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                "A required bundle file is missing, linked, or outside the bundle",
                f"/bundle/{relative_path}",
            )
    return None


def create_trial_bundle(
    document: Mapping[str, object],
    frozen_response: Mapping[str, object],
    bundle_dir: Path,
    *,
    trial_id: str,
    realization_record: bytes | None = None,
    proposal_record: bytes | None = None,
    composition_manifest: bytes | None = None,
) -> TrialBundleResult:
    """Create one self-contained bundle from an approved script and frozen response."""

    if not isinstance(trial_id, str) or not trial_id.strip():
        return _failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The trial ID must be a non-empty string",
            "/trial_id",
        )
    current_script = is_validated_script(document)
    if current_script:
        _, manifest_failure = _validated_composition_manifest(
            document,
            composition_manifest,
            trial_id=trial_id,
        )
        if manifest_failure is not None:
            return manifest_failure
    if realization_record is not None:
        try:
            record_value = json.loads(realization_record.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            return _failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                str(error),
                "/realization_record",
                trial_id=trial_id,
            )
        record_version = (
            record_value.get("schema_version") if isinstance(record_value, dict) else None
        )
        if record_version == 2:
            record_issue = verify_staged_realization_record(
                realization_record,
                document,
                frozen_response,
            )
            if record_issue is not None:
                return _failure(
                    record_issue.code,
                    record_issue.message,
                    record_issue.path,
                    trial_id=trial_id,
                )
        elif record_version != 1:
            return _failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                "The realization record schema version is not supported",
                "/realization_record/schema_version",
                trial_id=trial_id,
            )
    bundle = Path(bundle_dir)
    if bundle.exists():
        return _failure(
            IssueCode.STORAGE_CONFLICT,
            "The trial bundle directory already exists",
            "/bundle_dir",
            trial_id=trial_id,
        )
    staging: Path | None = None
    try:
        bundle.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".scoim-trial-", dir=bundle.parent))
        realization = realize_solo_piano_3m(
            document,
            frozen_response,
            staging / "artifacts",
        )
        document_validation = check_realization_script(document)
        script_record_name = (
            "validated_script"
            if document_validation.valid and current_script
            else "approved_script"
            if document_validation.valid
            else "script_input"
        )
        script_relative_path = (
            _VALIDATED_SCRIPT_PATH
            if script_record_name == "validated_script"
            else _FILE_PATHS["approved_script"]
            if script_record_name == "approved_script"
            else _REJECTED_SCRIPT_PATH
        )
        approved_path = staging / script_relative_path
        response_path = staging / _FILE_PATHS["frozen_response"]
        approved_path.write_bytes(_json_bytes(document))
        response_path.write_bytes(_json_bytes(frozen_response))
        if current_script:
            assert composition_manifest is not None
            (staging / _SOURCE_COMPOSITION_MANIFEST_PATH).write_bytes(composition_manifest)
        (staging / _FILE_PATHS["terminal"]).write_bytes(_json_bytes(realization.outcome.to_dict()))
        if realization_record is not None:
            model_record_path = staging / _MODEL_RUN_PATHS["realization_model_run"]
            model_record_path.parent.mkdir(exist_ok=True)
            model_record_path.write_bytes(realization_record)
        if proposal_record is not None:
            proposal_record_path = staging / _MODEL_RUN_PATHS["proposal_model_run"]
            proposal_record_path.parent.mkdir(exist_ok=True)
            proposal_record_path.write_bytes(proposal_record)
        script_hash = realization_script_sha256(document) if document_validation.valid else None
        files = {
            name: {
                "path": relative_path,
                "sha256": _sha256_file(staging / relative_path),
            }
            for name, relative_path in {**_FILE_PATHS, **_DIAGNOSTIC_FILE_PATHS}.items()
            if (staging / relative_path).is_file()
        }
        if script_record_name in {"validated_script", "script_input"}:
            files[script_record_name] = {
                "path": script_relative_path,
                "sha256": _sha256_file(staging / script_relative_path),
            }
        if current_script:
            files["source_composition_manifest"] = {
                "path": _SOURCE_COMPOSITION_MANIFEST_PATH,
                "sha256": _sha256_file(staging / _SOURCE_COMPOSITION_MANIFEST_PATH),
            }
        if realization_record is not None:
            relative_path = _MODEL_RUN_PATHS["realization_model_run"]
            files["realization_model_run"] = {
                "path": relative_path,
                "sha256": _sha256_file(staging / relative_path),
            }
        if proposal_record is not None:
            relative_path = _MODEL_RUN_PATHS["proposal_model_run"]
            files["proposal_model_run"] = {
                "path": relative_path,
                "sha256": _sha256_file(staging / relative_path),
            }
        response_version = frozen_response.get("schema_version")
        realizer_version = _REALIZER_VERSION_BY_RESPONSE_SCHEMA.get(
            response_version, _CURRENT_REALIZER_VERSION
        )
        script_reference = {
            "document_id": document.get("document_id"),
            "revision": document.get("revision"),
            "content_sha256": script_hash,
        }
        manifest = {
            "schema_version": (_CURRENT_BUNDLE_SCHEMA_VERSION if current_script else 3),
            "trial_id": trial_id,
            "realizer": {
                "package": "scoim",
                "package_version": _PACKAGE_VERSION,
                "realizer_version": realizer_version,
                "profile": _PROFILE,
            },
            "frozen_response_sha256": realization.frozen_response_sha256,
            "files": files,
        }
        if current_script:
            manifest.update(
                {
                    "bundle_type": _BUNDLE_TYPE,
                    "validated_script": script_reference,
                    "source_composition": {
                        "manifest_sha256": hashlib.sha256(composition_manifest).hexdigest()
                    },
                }
            )
        else:
            manifest["approved_script"] = script_reference
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        os.rename(staging, bundle)
        staging = None
    except OSError as error:
        return _failure(
            IssueCode.STORAGE_ERROR,
            str(error),
            "/bundle_dir",
            trial_id=trial_id,
        )
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    return TrialBundleResult(
        created=realization.realized,
        persisted=True,
        trial_path=bundle,
        outcome=realization.outcome,
        bundle_path=bundle if realization.realized else None,
        manifest_path=bundle / "manifest.json",
        trial_id=trial_id,
        frozen_response_sha256=realization.frozen_response_sha256,
    )


def create_failed_model_trial(
    document: Mapping[str, object],
    bundle_dir: Path,
    *,
    trial_id: str,
    outcome: ExecutionOutcome,
    realization_record: bytes,
    proposal_record: bytes | None = None,
    composition_manifest: bytes | None = None,
    realizer_version: int = _CURRENT_REALIZER_VERSION,
) -> TrialBundleResult:
    """Persist a started model trial that ended before a frozen response existed."""

    if not isinstance(trial_id, str) or not trial_id.strip():
        return _failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            "The trial ID must be a non-empty string",
            "/trial_id",
        )
    if (
        outcome.terminal_state is not TerminalState.FAILED
        or outcome.artifact_disposition is not ArtifactDisposition.NONE
    ):
        raise ValueError("a failed model trial requires a failed outcome without artifacts")
    if realizer_version not in _SUPPORTED_REALIZER_VERSIONS:
        raise ValueError("a failed model trial requires a supported realizer version")
    validation = check_realization_script(document)
    if not validation.valid:
        issue = validation.issues[0]
        return _failure(issue.code, issue.message, issue.path, trial_id=trial_id)
    current_script = is_validated_script(document)
    if current_script:
        _, manifest_failure = _validated_composition_manifest(
            document,
            composition_manifest,
            trial_id=trial_id,
        )
        if manifest_failure is not None:
            return manifest_failure
    bundle = Path(bundle_dir)
    if bundle.exists():
        return _failure(
            IssueCode.STORAGE_CONFLICT,
            "The trial bundle directory already exists",
            "/bundle_dir",
            trial_id=trial_id,
        )
    staging: Path | None = None
    try:
        bundle.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".scoim-trial-", dir=bundle.parent))
        script_relative_path = (
            _VALIDATED_SCRIPT_PATH if current_script else _FILE_PATHS["approved_script"]
        )
        approved_path = staging / script_relative_path
        terminal_path = staging / _FILE_PATHS["terminal"]
        model_record_path = staging / "model-runs" / "realization.json"
        model_record_path.parent.mkdir()
        approved_path.write_bytes(_json_bytes(document))
        terminal_path.write_bytes(_json_bytes(outcome.to_dict()))
        model_record_path.write_bytes(realization_record)
        if current_script:
            assert composition_manifest is not None
            (staging / _SOURCE_COMPOSITION_MANIFEST_PATH).write_bytes(composition_manifest)
        if proposal_record is not None:
            proposal_record_path = staging / _MODEL_RUN_PATHS["proposal_model_run"]
            proposal_record_path.write_bytes(proposal_record)
        script_hash = realization_script_sha256(document)
        script_record_name = "validated_script" if current_script else "approved_script"
        files = {
            script_record_name: {
                "path": script_relative_path,
                "sha256": _sha256_file(approved_path),
            },
            "terminal": {
                "path": _FILE_PATHS["terminal"],
                "sha256": _sha256_file(terminal_path),
            },
            "realization_model_run": {
                "path": "model-runs/realization.json",
                "sha256": _sha256_file(model_record_path),
            },
        }
        if proposal_record is not None:
            proposal_relative_path = _MODEL_RUN_PATHS["proposal_model_run"]
            files["proposal_model_run"] = {
                "path": proposal_relative_path,
                "sha256": _sha256_file(staging / proposal_relative_path),
            }
        if current_script:
            files["source_composition_manifest"] = {
                "path": _SOURCE_COMPOSITION_MANIFEST_PATH,
                "sha256": _sha256_file(staging / _SOURCE_COMPOSITION_MANIFEST_PATH),
            }
        manifest = {
            "schema_version": _CURRENT_BUNDLE_SCHEMA_VERSION if current_script else 3,
            "trial_id": trial_id,
            script_record_name: {
                "document_id": document["document_id"],
                "revision": document["revision"],
                "content_sha256": script_hash,
            },
            "realizer": {
                "package": "scoim",
                "package_version": _PACKAGE_VERSION,
                "realizer_version": realizer_version,
                "profile": _PROFILE,
            },
            "frozen_response_sha256": None,
            "files": files,
        }
        if current_script:
            assert composition_manifest is not None
            manifest.update(
                {
                    "bundle_type": _BUNDLE_TYPE,
                    "source_composition": {
                        "manifest_sha256": hashlib.sha256(composition_manifest).hexdigest()
                    },
                }
            )
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        os.rename(staging, bundle)
        staging = None
    except OSError as error:
        return _failure(
            IssueCode.STORAGE_ERROR,
            str(error),
            "/bundle_dir",
            trial_id=trial_id,
        )
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    return TrialBundleResult(
        created=False,
        persisted=True,
        trial_path=bundle,
        outcome=outcome,
        bundle_path=None,
        manifest_path=bundle / "manifest.json",
        trial_id=trial_id,
        frozen_response_sha256=None,
    )


def verify_failed_trial_bundle(bundle_dir: Path) -> ValidationIssue | None:
    """Verify a terminal failed trial that has no replayable artifacts."""

    bundle = Path(bundle_dir).resolve()
    try:
        if bundle.is_symlink() or not bundle.is_dir():
            raise ValueError("The failed trial bundle must be a real directory")
        actual_files: dict[str, Path] = {}
        for path in bundle.rglob("*"):
            relative = path.relative_to(bundle).as_posix()
            if path.is_symlink():
                raise ValueError(f"A failed trial bundle entry is linked: {relative}")
            if path.is_file():
                actual_files[relative] = path
        manifest = _load_json_object(bundle / "manifest.json", "/manifest")
        if isinstance(manifest, TrialReplayResult):
            raise ValueError(manifest.issues[0].message)
        current = manifest.get("schema_version") == _CURRENT_BUNDLE_SCHEMA_VERSION
        script_record_name = "validated_script" if current else "approved_script"
        script_path = _VALIDATED_SCRIPT_PATH if current else _FILE_PATHS["approved_script"]
        document = _load_json_object(bundle / script_path, f"/{script_record_name}")
        if isinstance(document, TrialReplayResult):
            raise ValueError(document.issues[0].message)
        terminal_value = _load_json_object(bundle / _FILE_PATHS["terminal"], "/terminal")
        if isinstance(terminal_value, TrialReplayResult):
            raise ValueError(terminal_value.issues[0].message)
        ExecutionOutcome.from_dict(terminal_value)
        expected_manifest_fields = _CURRENT_MANIFEST_FIELDS if current else _MANIFEST_FIELDS
        if (
            set(manifest) != expected_manifest_fields
            or manifest.get("frozen_response_sha256") is not None
            or (current and manifest.get("bundle_type") != _BUNDLE_TYPE)
        ):
            raise ValueError("The failed trial manifest fields are invalid")
        realizer = _object(manifest.get("realizer"))
        if realizer is None or realizer.get("realizer_version") != 5:
            raise ValueError("The failed staged trial must use realizer version 5")
        file_records = _object(manifest.get("files"))
        if file_records is None:
            raise ValueError("The failed trial has no file records")
        required_names = {script_record_name, "terminal", "realization_model_run"}
        if current:
            required_names.add("source_composition_manifest")
        if not required_names.issubset(file_records) or not set(file_records).issubset(
            required_names | {"proposal_model_run"}
        ):
            raise ValueError("The failed trial file records are invalid")
        expected_files = {"manifest.json"}
        for name, raw_record in file_records.items():
            record = _object(raw_record)
            if record is None or set(record) != {"path", "sha256"}:
                raise ValueError(f"A failed trial file record is invalid: {name}")
            relative = record["path"]
            if not isinstance(relative, str) or relative not in actual_files:
                raise ValueError(f"A failed trial file is missing: {name}")
            if _sha256_file(actual_files[relative]) != record["sha256"]:
                raise ValueError(f"A failed trial file hash does not match: {name}")
            expected_files.add(relative)
        if set(actual_files) != expected_files:
            raise ValueError("The failed trial bundle contains an unrecorded file")
        validation = check_realization_script(document)
        approved = _object(manifest.get(script_record_name))
        if not validation.valid or approved != {
            "document_id": document.get("document_id"),
            "revision": document.get("revision"),
            "content_sha256": realization_script_sha256(document),
        }:
            raise ValueError("The failed trial approved script does not match its manifest")
        if current:
            source_manifest_path = actual_files[_SOURCE_COMPOSITION_MANIFEST_PATH]
            source_manifest = _load_json_object(
                source_manifest_path,
                "/source_composition_manifest",
            )
            if isinstance(source_manifest, TrialReplayResult):
                raise ValueError(source_manifest.issues[0].message)
            source_reference = _object(manifest.get("source_composition"))
            if (
                source_reference is None
                or source_reference.get("manifest_sha256") != _sha256_file(source_manifest_path)
                or source_manifest.get("bundle_type") != "composition"
                or source_manifest.get("schema_version") != 2
                or source_manifest.get("terminal_state") != "succeeded"
                or source_manifest.get("composition_id") != document.get("document_id")
                or source_manifest.get("validated_script_content_sha256")
                != realization_script_sha256(document)
            ):
                raise ValueError("The failed trial composition lineage does not match its script")
        realization_record = cast(dict[str, object], file_records["realization_model_run"])
        record_path = actual_files[cast(str, realization_record["path"])]
        record_issue = verify_failed_staged_realization_record(record_path.read_bytes(), document)
        if record_issue is not None:
            return record_issue
        realization_value = _load_json_object(record_path, "/realization_record")
        if isinstance(realization_value, TrialReplayResult):
            raise ValueError(realization_value.issues[0].message)
        if realization_value.get("issues") != terminal_value.get("issues"):
            raise ValueError("The failed trial issues do not match its realization record")
    except (
        KeyError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        return ValidationIssue(IssueCode.MODEL_OUTPUT_INVALID, str(error), "/bundle")
    return None


def replay_trial_bundle(bundle_dir: Path, output_dir: Path) -> TrialReplayResult:
    """Replay a self-contained trial bundle without an external model call."""

    bundle = Path(bundle_dir)
    output = Path(output_dir)
    bundle_resolved = bundle.resolve()
    output_resolved = output.resolve()
    if (
        output.exists()
        or output_resolved == bundle_resolved
        or bundle_resolved in output_resolved.parents
    ):
        return _replay_failure(
            IssueCode.STORAGE_CONFLICT,
            "The replay output must be a new directory outside the trial bundle",
            "/output_dir",
        )
    wrapper: Path | None = None
    manifest_value = _load_json_object(bundle / "manifest.json", "/manifest")
    if isinstance(manifest_value, TrialReplayResult):
        return manifest_value
    trial_id, manifest_failure = _manifest_issue(manifest_value)
    if manifest_failure is not None:
        return manifest_failure
    assert trial_id is not None
    current = manifest_value["schema_version"] == _CURRENT_BUNDLE_SCHEMA_VERSION
    layout_failure = _bundle_layout_issue(bundle, validated_script=current)
    if layout_failure is not None:
        return layout_failure
    script_path = _VALIDATED_SCRIPT_PATH if current else _FILE_PATHS["approved_script"]
    script_issue_path = "/validated_script" if current else "/approved_script"
    document_value = _load_json_object(bundle / script_path, script_issue_path)
    if isinstance(document_value, TrialReplayResult):
        return document_value
    response_value = _load_json_object(bundle / _FILE_PATHS["frozen_response"], "/frozen_response")
    if isinstance(response_value, TrialReplayResult):
        return response_value
    terminal_value = _load_json_object(bundle / _FILE_PATHS["terminal"], "/terminal")
    if isinstance(terminal_value, TrialReplayResult):
        return terminal_value
    manifest = manifest_value
    document = document_value
    frozen_response = response_value
    try:
        saved_outcome = ExecutionOutcome.from_dict(terminal_value)
    except ValueError as error:
        return _replay_failure(
            IssueCode.MODEL_OUTPUT_INVALID,
            str(error),
            "/terminal",
        )
    try:
        file_records = cast(dict[str, dict[str, object]], manifest["files"])
        for name, record in file_records.items():
            relative_path = cast(str, record["path"])
            saved_path = bundle / relative_path
            if (
                saved_path.is_symlink()
                or not saved_path.is_file()
                or not saved_path.resolve().is_relative_to(bundle_resolved)
            ):
                return _replay_failure(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "A recorded bundle file is missing, linked, or outside the bundle",
                    f"/files/{name}",
                    trial_id=trial_id,
                )
            if _sha256_file(saved_path) != record["sha256"]:
                return _replay_failure(
                    IssueCode.LINEAGE_MISMATCH,
                    "A bundled file hash does not match the manifest",
                    f"/files/{name}/sha256",
                    trial_id=trial_id,
                )
        if current:
            source_manifest_path = bundle / _SOURCE_COMPOSITION_MANIFEST_PATH
            source_manifest_sha256 = _sha256_file(source_manifest_path)
            source_reference = cast(dict[str, object], manifest["source_composition"])
            if source_manifest_sha256 != source_reference["manifest_sha256"]:
                return _replay_failure(
                    IssueCode.LINEAGE_MISMATCH,
                    "The source composition manifest hash does not match",
                    "/manifest/source_composition/manifest_sha256",
                    trial_id=trial_id,
                )
            source_manifest_value = _load_json_object(
                source_manifest_path, "/source_composition_manifest"
            )
            if isinstance(source_manifest_value, TrialReplayResult):
                return source_manifest_value
            if (
                source_manifest_value.get("bundle_type") != "composition"
                or source_manifest_value.get("schema_version") != 2
                or source_manifest_value.get("terminal_state") != "succeeded"
                or source_manifest_value.get("composition_id") != document.get("document_id")
                or source_manifest_value.get("validated_script_content_sha256")
                != realization_script_sha256(document)
            ):
                return _replay_failure(
                    IssueCode.LINEAGE_MISMATCH,
                    "The source composition manifest does not identify the bundled script",
                    "/source_composition_manifest",
                    trial_id=trial_id,
                )
        validation = check_realization_script(document)
        if not validation.valid:
            issue = validation.issues[0]
            return _replay_failure(
                IssueCode.LINEAGE_MISMATCH,
                issue.message,
                f"{script_issue_path}{issue.path}",
                trial_id=trial_id,
            )
        script_reference_field = "validated_script" if current else "approved_script"
        approved_reference = cast(dict[str, object], manifest[script_reference_field])
        approved_values = {
            "document_id": document["document_id"],
            "revision": document["revision"],
            "content_sha256": realization_script_sha256(document),
        }
        for field, actual in approved_values.items():
            if approved_reference[field] != actual:
                return _replay_failure(
                    IssueCode.LINEAGE_MISMATCH,
                    "The approved script does not match its manifest reference",
                    f"/manifest/{script_reference_field}/{field}",
                    trial_id=trial_id,
                )
        realization_record_entry = file_records.get("realization_model_run")
        if realization_record_entry is not None:
            realization_record_path = bundle / cast(str, realization_record_entry["path"])
            realization_record = realization_record_path.read_bytes()
            realization_record_value = json.loads(realization_record.decode("utf-8"))
            realization_record_version = (
                realization_record_value.get("schema_version")
                if isinstance(realization_record_value, dict)
                else None
            )
            if realization_record_version == 2:
                record_issue = verify_staged_realization_record(
                    realization_record,
                    document,
                    frozen_response,
                )
                if record_issue is not None:
                    return _replay_failure(
                        record_issue.code,
                        record_issue.message,
                        record_issue.path,
                        trial_id=trial_id,
                    )
            elif realization_record_version != 1:
                return _replay_failure(
                    IssueCode.MODEL_OUTPUT_INVALID,
                    "The realization record schema version is not supported",
                    "/realization_record/schema_version",
                    trial_id=trial_id,
                )
        try:
            response_sha256 = hashlib.sha256(rfc8785.dumps(frozen_response)).hexdigest()
        except (TypeError, ValueError):
            return _replay_failure(
                IssueCode.MODEL_OUTPUT_INVALID,
                "The frozen response is not canonical JSON data",
                "/frozen_response",
                trial_id=trial_id,
            )
        if response_sha256 != manifest["frozen_response_sha256"]:
            return _replay_failure(
                IssueCode.LINEAGE_MISMATCH,
                "The frozen response content hash does not match the manifest",
                "/frozen_response_sha256",
                trial_id=trial_id,
            )
        realizer = cast(dict[str, object], manifest["realizer"])
        expected_response_version = _RESPONSE_SCHEMA_BY_REALIZER_VERSION[
            cast(int, realizer["realizer_version"])
        ]
        if frozen_response.get("schema_version") != expected_response_version:
            return _replay_failure(
                IssueCode.LINEAGE_MISMATCH,
                "The frozen response schema does not match the saved realizer version",
                "/manifest/realizer/realizer_version",
                trial_id=trial_id,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        wrapper = Path(tempfile.mkdtemp(prefix=".scoim-replay-", dir=output.parent))
        replayed = realize_solo_piano_3m(document, frozen_response, wrapper / "artifacts")
        if not replayed.realized:
            return TrialReplayResult(
                False,
                replayed.outcome,
                None,
                trial_id,
                {},
            )
        if replayed.outcome != saved_outcome:
            return _replay_failure(
                IssueCode.LINEAGE_MISMATCH,
                "The replayed terminal outcome does not match the bundle",
                "/terminal",
                trial_id=trial_id,
            )
        artifact_sha256 = {
            name: _sha256_file(wrapper / "artifacts" / Path(relative_path).name)
            for name, relative_path in {**_FILE_PATHS, **_DIAGNOSTIC_FILE_PATHS}.items()
            if name in {"musicxml", "smf", "diagnostics"} | set(_DIAGNOSTIC_FILE_PATHS)
            and name in file_records
        }
        for name, actual in artifact_sha256.items():
            if actual != manifest["files"][name]["sha256"]:
                return _replay_failure(
                    IssueCode.LINEAGE_MISMATCH,
                    "A replayed artifact hash does not match the manifest",
                    f"/files/{name}/sha256",
                    trial_id=trial_id,
                )
        os.rename(wrapper / "artifacts", output)
    except OSError as error:
        return _replay_failure(IssueCode.STORAGE_ERROR, str(error), "/bundle_dir")
    finally:
        if wrapper is not None:
            shutil.rmtree(wrapper, ignore_errors=True)
    return TrialReplayResult(
        replayed=True,
        outcome=saved_outcome,
        output_path=output,
        trial_id=trial_id,
        artifact_sha256=artifact_sha256,
    )
