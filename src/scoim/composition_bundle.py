"""Atomic, self-contained composition bundles for SCoIM phase 2."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from llm_musical_composer.run_state import atomic_write_json, sha256_file

from .flow_validation import check_flow, flow_content_sha256
from .proposal import ProposalRunner
from .script_compilation import CompilationRequest, compile_script
from .script_validation import check_script_document, script_content_sha256
from .validation import CheckResult, IssueCode, ValidationIssue

_BUNDLE_TYPE = "composition"
_BUNDLE_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class CompositionResult:
    """Terminal state of creating one immutable composition bundle."""

    succeeded: bool
    persisted: bool
    bundle_dir: Path | None
    issues: tuple[ValidationIssue, ...]


def compose_flow(
    request: CompilationRequest,
    runner: ProposalRunner,
    destination: str | Path,
) -> CompositionResult:
    """Compile one approved flow and atomically publish its complete evidence."""
    target = Path(destination)
    if target.exists():
        return CompositionResult(
            False,
            False,
            None,
            (
                ValidationIssue(
                    IssueCode.STORAGE_CONFLICT,
                    "The composition bundle already exists",
                    "/destination",
                ),
            ),
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    bundle = temporary_root / "bundle"
    bundle.mkdir()
    try:
        compilation = compile_script(request, runner, temporary_root / "run")
        if compilation.run_dir is None:
            return CompositionResult(False, False, None, compilation.issues)
        (bundle / "model-runs").mkdir()
        shutil.copytree(
            compilation.run_dir,
            bundle / "model-runs" / "script-compilation",
        )
        atomic_write_json(bundle / "approved-flow.json", dict(request.approved_flow))
        atomic_write_json(bundle / "projection-targets.json", list(compilation.projection_targets))
        atomic_write_json(bundle / "model-runs" / "compilation.json", compilation.response_record)
        state = "succeeded" if compilation.compiled else "failed"
        terminal: dict[str, object] = {
            "schema_version": 1,
            "composition_id": request.composition_id,
            "state": state,
            "issues": [
                {"code": issue.code.value, "message": issue.message, "path": issue.path}
                for issue in compilation.issues
            ],
        }
        if compilation.metrics is not None:
            terminal["downstream_call_budget"] = asdict(compilation.metrics)
        atomic_write_json(bundle / "terminal.json", terminal)
        if compilation.document is not None:
            atomic_write_json(bundle / "validated-script.json", compilation.document)
        file_hashes = {
            path.relative_to(bundle).as_posix(): sha256_file(path)
            for path in sorted(bundle.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, object] = {
            "bundle_type": _BUNDLE_TYPE,
            "schema_version": _BUNDLE_SCHEMA_VERSION,
            "composition_id": request.composition_id,
            "terminal_state": state,
            "approved_flow_content_sha256": flow_content_sha256(request.approved_flow),
            "files": file_hashes,
        }
        if compilation.document is not None:
            manifest["validated_script_content_sha256"] = script_content_sha256(
                compilation.document
            )
        atomic_write_json(bundle / "manifest.json", manifest)
        verification = verify_composition_bundle(bundle)
        if not verification.valid:
            return CompositionResult(False, False, None, verification.issues)
        os.replace(bundle, target)
        return CompositionResult(compilation.compiled, True, target, compilation.issues)
    except OSError as error:
        return CompositionResult(
            False,
            False,
            None,
            (ValidationIssue(IssueCode.STORAGE_ERROR, str(error), "/destination"),),
        )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def verify_composition_bundle(bundle_dir: str | Path) -> CheckResult:
    """Verify a copied bundle without network access or its original RunStore."""
    root = Path(bundle_dir)
    try:
        manifest = _read_object(root / "manifest.json")
        terminal = _read_object(root / "terminal.json")
        flow = _read_object(root / "approved-flow.json")
        targets = json.loads((root / "projection-targets.json").read_text(encoding="utf-8"))
        compilation = _read_object(root / "model-runs" / "compilation.json")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        return CheckResult(
            False, (ValidationIssue(IssueCode.STORAGE_ERROR, str(error), str(root)),)
        )
    issues: list[ValidationIssue] = []
    state = terminal.get("state")
    expected_manifest_fields = {
        "bundle_type",
        "schema_version",
        "composition_id",
        "terminal_state",
        "approved_flow_content_sha256",
        "files",
    }
    if state == "succeeded":
        expected_manifest_fields.add("validated_script_content_sha256")
    if set(manifest) != expected_manifest_fields:
        issues.append(
            ValidationIssue(
                IssueCode.SCHEMA_INVALID,
                "The composition manifest fields do not match the bundle contract",
                "/manifest",
            )
        )
    if (
        manifest.get("bundle_type") != _BUNDLE_TYPE
        or manifest.get("schema_version") != _BUNDLE_SCHEMA_VERSION
    ):
        issues.append(
            ValidationIssue(
                IssueCode.UNSUPPORTED_SCHEMA_VERSION,
                "The composition bundle type or schema version is unsupported",
                "/manifest/schema_version",
            )
        )
    if not isinstance(targets, list) or not isinstance(compilation, dict):
        issues.append(
            ValidationIssue(IssueCode.SCHEMA_INVALID, "Bundle evidence has an invalid shape", "/")
        )
    if state not in {"succeeded", "failed"} or manifest.get("terminal_state") != state:
        issues.append(
            ValidationIssue(
                IssueCode.SCHEMA_INVALID, "Bundle terminal states do not agree", "/terminal_state"
            )
        )
    script_path = root / "validated-script.json"
    if (state == "succeeded") != script_path.is_file():
        issues.append(
            ValidationIssue(
                IssueCode.SCHEMA_INVALID,
                "Only a successful bundle may contain validated-script.json",
                "/validated-script.json",
            )
        )
    file_hashes = manifest.get("files")
    if not isinstance(file_hashes, dict):
        issues.append(
            ValidationIssue(IssueCode.SCHEMA_INVALID, "Manifest files must be an object", "/files")
        )
    else:
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        if set(file_hashes) != actual_paths:
            issues.append(
                ValidationIssue(
                    IssueCode.STORAGE_ERROR,
                    "Manifest file inventory does not match the bundle",
                    "/files",
                )
            )
        for relative, expected in file_hashes.items():
            path = root / cast(str, relative)
            if not path.is_file() or sha256_file(path) != expected:
                issues.append(
                    ValidationIssue(
                        IssueCode.STORAGE_ERROR,
                        "A fixed bundle file hash does not match",
                        f"/files/{relative}",
                    )
                )
    flow_check = check_flow(flow)
    issues.extend(flow_check.issues)
    if flow_check.valid and manifest.get("approved_flow_content_sha256") != flow_content_sha256(
        flow
    ):
        issues.append(
            ValidationIssue(
                IssueCode.LINEAGE_MISMATCH,
                "The approved flow content hash does not match",
                "/approved_flow_content_sha256",
            )
        )
    if state == "succeeded" and script_path.is_file():
        try:
            script = _read_object(script_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            issues.append(
                ValidationIssue(IssueCode.STORAGE_ERROR, str(error), "/validated-script.json")
            )
        else:
            script_check = check_script_document(script)
            issues.extend(script_check.issues)
            source = script.get("source_flow")
            if isinstance(source, dict) and (
                source.get("document_id") != flow.get("document_id")
                or source.get("revision") != flow.get("revision")
                or source.get("content_sha256") != flow_content_sha256(flow)
            ):
                issues.append(
                    ValidationIssue(
                        IssueCode.LINEAGE_MISMATCH,
                        "Validated script lineage does not match the bundled flow",
                        "/validated-script.json/source_flow",
                    )
                )
            if manifest.get("validated_script_content_sha256") != script_content_sha256(script):
                issues.append(
                    ValidationIssue(
                        IssueCode.LINEAGE_MISMATCH,
                        "Validated script content hash does not match",
                        "/validated_script_content_sha256",
                    )
                )
    return CheckResult(not issues, tuple(issues))


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return cast(dict[str, object], value)
