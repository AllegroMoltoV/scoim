"""Immutable composition bundles for the solo-piano v2 generation path."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from llm_musical_composer.run_state import atomic_write_json, sha256_file

from .finite_model_operation import check_finite_model_operation_records
from .flow_validation import check_flow, flow_content_sha256
from .phase3_realization import expected_phase2_projection_targets
from .projection_ledger import (
    ProjectionLedgerEntry,
    validate_expected_projection_targets,
    validate_projection_ledger,
)
from .script_0_4_validation import (
    check_script_0_4_document,
    script_0_4_content_sha256,
)
from .validation import CheckResult, IssueCode, ValidationIssue

_BUNDLE_TYPE = "composition"
_BUNDLE_SCHEMA_VERSION = 3
_TARGET_PROFILE = "solo_piano_3m_v2"


@dataclass(frozen=True, slots=True)
class V2CompositionBundleResult:
    """Result of publishing one v2 composition bundle."""

    created: bool
    bundle_dir: Path | None
    issues: tuple[ValidationIssue, ...]


def create_v2_composition_bundle(
    phase2_run_dir: str | Path,
    destination: str | Path,
) -> V2CompositionBundleResult:
    """Package one complete phase-2 run without another model call."""
    source = Path(phase2_run_dir).resolve()
    target = Path(destination).resolve()
    if target.exists():
        return _failure(IssueCode.STORAGE_CONFLICT, "bundle already exists", target)
    operation_records = check_finite_model_operation_records(source)
    if not operation_records.valid:
        return _failure(
            IssueCode.LINEAGE_MISMATCH,
            f"phase2 model operation record is invalid: {operation_records.issues[0].message}",
            source,
        )
    try:
        spec = _read_object(source / "run-spec.json")
        flow = _read_object(source / "inputs" / "approved-flow.json")
        script = _read_object(source / "outputs" / "validated-script.json")
        summary = _read_object(source / "outputs" / "compilation.json")
        ledger_values = _read_array(source / "outputs" / "projection-ledger.json")
        ledger = tuple(
            ProjectionLedgerEntry(**cast(dict[str, object], value)) for value in ledger_values
        )
        _validate_phase2(spec, flow, script, summary, ledger)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _failure(IssueCode.SEMANTIC_INVALID, str(error), source)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    bundle = temporary_root / "bundle"
    try:
        bundle.mkdir()
        shutil.copy2(source / "inputs" / "approved-flow.json", bundle / "approved-flow.json")
        shutil.copy2(
            source / "outputs" / "validated-script.json",
            bundle / "validated-script.json",
        )
        shutil.copy2(
            source / "outputs" / "projection-ledger.json",
            bundle / "projection-ledger.json",
        )
        shutil.copytree(source, bundle / "model-runs" / "phase2")
        atomic_write_json(
            bundle / "terminal.json",
            {
                "outcome": "complete",
                "composition_id": spec["composition_id"],
                "issues": [],
            },
        )
        file_hashes = {
            path.relative_to(bundle).as_posix(): sha256_file(path)
            for path in sorted(bundle.rglob("*"))
            if path.is_file()
        }
        atomic_write_json(
            bundle / "manifest.json",
            {
                "bundle_type": _BUNDLE_TYPE,
                "schema_version": _BUNDLE_SCHEMA_VERSION,
                "target_profile": _TARGET_PROFILE,
                "composition_id": spec["composition_id"],
                "approved_flow_content_sha256": flow_content_sha256(flow),
                "validated_script_content_sha256": script_0_4_content_sha256(script),
                "files": file_hashes,
            },
        )
        verification = verify_v2_composition_bundle(bundle)
        if not verification.valid:
            return V2CompositionBundleResult(False, None, verification.issues)
        os.replace(bundle, target)
        return V2CompositionBundleResult(True, target, ())
    except OSError as error:
        return _failure(IssueCode.STORAGE_ERROR, str(error), target)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def verify_v2_composition_bundle(bundle_dir: str | Path) -> CheckResult:
    """Verify one copied v2 composition bundle without its source run."""
    root = Path(bundle_dir).resolve()
    try:
        manifest = _read_object(root / "manifest.json")
        terminal = _read_object(root / "terminal.json")
        flow = _read_object(root / "approved-flow.json")
        script = _read_object(root / "validated-script.json")
        ledger_values = _read_array(root / "projection-ledger.json")
        ledger = tuple(
            ProjectionLedgerEntry(**cast(dict[str, object], value)) for value in ledger_values
        )
        phase2 = root / "model-runs" / "phase2"
        operation_records = check_finite_model_operation_records(phase2)
        if not operation_records.valid:
            raise ValueError(
                f"phase2 model operation record is invalid: {operation_records.issues[0].message}"
            )
        spec = _read_object(phase2 / "run-spec.json")
        summary = _read_object(phase2 / "outputs" / "compilation.json")
        _validate_phase2(spec, flow, script, summary, ledger)
        _verify_manifest(root, manifest, terminal, spec, flow, script)
        if (root / "approved-flow.json").read_bytes() != (
            phase2 / "inputs" / "approved-flow.json"
        ).read_bytes():
            raise ValueError("bundled flow differs from the phase-2 input")
        if (root / "validated-script.json").read_bytes() != (
            phase2 / "outputs" / "validated-script.json"
        ).read_bytes():
            raise ValueError("bundled script differs from the phase-2 output")
        if (root / "projection-ledger.json").read_bytes() != (
            phase2 / "outputs" / "projection-ledger.json"
        ).read_bytes():
            raise ValueError("bundled projection ledger differs from the phase-2 output")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return CheckResult(
            False,
            (ValidationIssue(IssueCode.LINEAGE_MISMATCH, str(error), "/bundle"),),
        )
    return CheckResult(True, ())


def _verify_manifest(
    root: Path,
    manifest: Mapping[str, object],
    terminal: Mapping[str, object],
    spec: Mapping[str, object],
    flow: Mapping[str, object],
    script: Mapping[str, object],
) -> None:
    expected_fields = {
        "bundle_type",
        "schema_version",
        "target_profile",
        "composition_id",
        "approved_flow_content_sha256",
        "validated_script_content_sha256",
        "files",
    }
    if set(manifest) != expected_fields:
        raise ValueError("v2 composition manifest fields are invalid")
    if (
        manifest.get("bundle_type") != _BUNDLE_TYPE
        or manifest.get("schema_version") != _BUNDLE_SCHEMA_VERSION
        or manifest.get("target_profile") != _TARGET_PROFILE
        or manifest.get("composition_id") != spec.get("composition_id")
        or terminal.get("outcome") != "complete"
        or terminal.get("composition_id") != spec.get("composition_id")
    ):
        raise ValueError("v2 composition bundle identity is invalid")
    if manifest.get("approved_flow_content_sha256") != flow_content_sha256(flow):
        raise ValueError("approved flow content hash differs")
    if manifest.get("validated_script_content_sha256") != script_0_4_content_sha256(script):
        raise ValueError("validated script content hash differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or any(
        not isinstance(relative, str) or not isinstance(digest, str)
        for relative, digest in files.items()
    ):
        raise ValueError("v2 composition file hashes are invalid")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(files) != actual_paths:
        raise ValueError("v2 composition file inventory differs")
    for relative, digest in cast(dict[str, str], files).items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"v2 composition file hash differs: {relative}")


def _validate_phase2(
    spec: Mapping[str, object],
    flow: Mapping[str, object],
    script: Mapping[str, object],
    summary: Mapping[str, object],
    ledger: tuple[ProjectionLedgerEntry, ...],
) -> None:
    if (
        spec.get("operation") != "script-0.4.0-compilation"
        or spec.get("target_profile") != _TARGET_PROFILE
        or summary.get("outcome") != "complete"
    ):
        raise ValueError("phase-2 run is not complete v2 output")
    composition_id = spec.get("composition_id")
    if not isinstance(composition_id, str) or not composition_id:
        raise ValueError("phase-2 composition ID is invalid")
    flow_check = check_flow(flow)
    if not flow_check.valid or flow.get("status") != "approved":
        raise ValueError("phase-2 approved flow is invalid")
    script_check = check_script_0_4_document(script)
    if not script_check.valid:
        raise ValueError("phase-2 validated script is invalid")
    source_flow = cast(Mapping[str, object], script["source_flow"])
    if (
        source_flow.get("document_id") != flow.get("document_id")
        or source_flow.get("revision") != flow.get("revision")
        or source_flow.get("content_sha256") != flow_content_sha256(flow)
    ):
        raise ValueError("phase-2 script lineage does not match its flow")
    validate_projection_ledger(ledger)
    validate_expected_projection_targets(ledger, expected_phase2_projection_targets(script))


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return cast(dict[str, object], value)


def _read_array(path: Path) -> list[object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"expected a JSON object array: {path}")
    return cast(list[object], value)


def _failure(code: IssueCode, message: str, path: Path) -> V2CompositionBundleResult:
    return V2CompositionBundleResult(
        False,
        None,
        (ValidationIssue(code, message, str(path)),),
    )
