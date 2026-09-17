"""Final v2 artifact bundle creation and model-free replay."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from llm_musical_composer.run_state import (
    atomic_write_bytes,
    atomic_write_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
)

from .finite_model_operation import check_finite_model_operation_records
from .phase7_state import load_complete_phase7_run
from .score_rendering import (
    check_rendered_performance_smf,
    check_score_musicxml,
    write_rendered_performance_smf,
    write_score_musicxml,
)
from .validation import CheckResult, IssueCode, ValidationIssue

_PHASE_NAMES = ("phase3", "phase4", "phase5", "phase6", "phase7")


@dataclass(frozen=True, slots=True)
class Phase8BundleRequest:
    phase7_run_dir: str | Path
    phase_run_dirs: Mapping[str, str | Path]
    composition_id: str
    trial_id: str
    composition_manifest: bytes


@dataclass(frozen=True, slots=True)
class Phase8BundleResult:
    created: bool
    bundle_dir: Path | None
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True, slots=True)
class Phase8ReplayResult:
    replayed: bool
    output_dir: Path | None
    issues: tuple[ValidationIssue, ...]


def verify_phase8_bundle(bundle_dir: str | Path) -> CheckResult:
    """Verify one final v2 bundle without creating replay artifacts."""
    issues = _verify_bundle(Path(bundle_dir).resolve())
    return CheckResult(not issues, issues)


def create_phase8_bundle(
    request: Phase8BundleRequest, destination: str | Path
) -> Phase8BundleResult:
    """Publish one self-contained bundle after artifact round-trip checks pass."""
    target = Path(destination).resolve()
    if target.exists():
        return _bundle_failure(IssueCode.STORAGE_CONFLICT, "bundle already exists", target)
    if set(request.phase_run_dirs) != set(_PHASE_NAMES[:-1]):
        return _bundle_failure(
            IssueCode.SEMANTIC_INVALID,
            "phase 3 through phase 6 run directories are required",
            target,
        )
    if not isinstance(request.composition_manifest, bytes):
        return _bundle_failure(
            IssueCode.LINEAGE_MISMATCH,
            "composition manifest must be bytes",
            target,
        )
    try:
        composition_manifest = json.loads(request.composition_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return _bundle_failure(IssueCode.SEMANTIC_INVALID, str(error), target)
    if (
        not isinstance(request.composition_id, str)
        or not request.composition_id.strip()
        or not isinstance(request.trial_id, str)
        or not request.trial_id.strip()
        or not isinstance(composition_manifest, dict)
        or composition_manifest.get("bundle_type") != "composition"
        or composition_manifest.get("schema_version") != 3
        or composition_manifest.get("target_profile") != "solo_piano_3m_v2"
        or composition_manifest.get("composition_id") != request.composition_id
    ):
        return _bundle_failure(
            IssueCode.LINEAGE_MISMATCH,
            "composition manifest identity is invalid",
            target,
        )
    phase_dirs = {
        **{name: Path(path).resolve() for name, path in request.phase_run_dirs.items()},
        "phase7": Path(request.phase7_run_dir).resolve(),
    }
    if any(not path.is_dir() for path in phase_dirs.values()):
        return _bundle_failure(IssueCode.STORAGE_ERROR, "a phase run directory is missing", target)
    for phase_name, phase_dir in phase_dirs.items():
        operation_records = check_finite_model_operation_records(phase_dir)
        if not operation_records.valid:
            return _bundle_failure(
                IssueCode.LINEAGE_MISMATCH,
                f"{phase_name} model operation record is invalid: "
                f"{operation_records.issues[0].message}",
                target,
            )
    try:
        loaded = load_complete_phase7_run(phase_dirs["phase7"])
    except (KeyError, OSError, TypeError, ValueError) as error:
        return _bundle_failure(IssueCode.SEMANTIC_INVALID, str(error), target)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent)).resolve()
    bundle = temporary_root / "bundle"
    try:
        bundle.mkdir()
        model_runs = bundle / "model-runs"
        model_runs.mkdir()
        for phase_name in _PHASE_NAMES:
            shutil.copytree(phase_dirs[phase_name], model_runs / phase_name)

        inputs = bundle / "inputs"
        inputs.mkdir()
        atomic_write_json(inputs / "validated-script.json", dict(loaded.validated_script))
        atomic_write_json(inputs / "piece-plan.json", asdict(loaded.plan))
        atomic_write_json(inputs / "score-spec.json", asdict(loaded.score))
        atomic_write_json(inputs / "performance-spec.json", asdict(loaded.performance))
        atomic_write_json(inputs / "rendered-performance.json", asdict(loaded.rendered))
        atomic_write_json(
            inputs / "projection-ledger.json",
            [asdict(entry) for entry in loaded.cumulative_projection_ledger],
        )

        lineage = bundle / "lineage"
        lineage.mkdir()
        atomic_write_bytes(
            lineage / "composition-manifest.json",
            request.composition_manifest,
        )

        artifacts = bundle / "artifacts"
        artifacts.mkdir()
        musicxml_path = artifacts / "score.musicxml"
        smf_path = artifacts / "final.mid"
        write_score_musicxml(loaded.validated_script, loaded.plan, loaded.score, musicxml_path)
        write_rendered_performance_smf(loaded.rendered, smf_path)
        checks = {
            "musicxml": check_score_musicxml(
                loaded.validated_script, loaded.plan, loaded.score, musicxml_path
            ),
            "smf": check_rendered_performance_smf(loaded.rendered, smf_path),
        }
        if any(check["status"] != "passed" for check in checks.values()):
            raise ValueError("a final artifact round-trip check failed")
        atomic_write_json(bundle / "checks.json", checks)
        atomic_write_json(
            bundle / "terminal.json",
            {"outcome": "complete", "issues": []},
        )
        file_hashes = {
            path.relative_to(bundle).as_posix(): sha256_file(path)
            for path in sorted(bundle.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "bundle_type": "scoim-generation-trial",
            "schema_version": 2,
            "target_profile": "solo_piano_3m_v2",
            "composition_id": request.composition_id,
            "trial_id": request.trial_id,
            "composition_manifest_sha256": sha256_bytes(request.composition_manifest),
            "validated_script_sha256": sha256_json(loaded.validated_script),
            "piece_plan_sha256": sha256_json(asdict(loaded.plan)),
            "score_spec_sha256": sha256_json(asdict(loaded.score)),
            "performance_spec_sha256": sha256_json(asdict(loaded.performance)),
            "rendered_performance_sha256": sha256_json(asdict(loaded.rendered)),
            "files": file_hashes,
        }
        atomic_write_json(bundle / "manifest.json", manifest)
        verification = _verify_bundle(bundle)
        if verification:
            return Phase8BundleResult(False, None, verification)
        os.replace(bundle, target)
        return Phase8BundleResult(True, target, ())
    except (OSError, TypeError, ValueError) as error:
        return _bundle_failure(IssueCode.STORAGE_ERROR, str(error), target)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def replay_phase8_bundle(bundle_dir: str | Path, destination: str | Path) -> Phase8ReplayResult:
    """Regenerate MusicXML and SMF from a verified bundle without model access."""
    bundle = Path(bundle_dir).resolve()
    target = Path(destination).resolve()
    if target.exists():
        return _replay_failure(IssueCode.STORAGE_CONFLICT, "replay output exists")
    issues = _verify_bundle(bundle)
    if issues:
        return Phase8ReplayResult(False, None, issues)
    try:
        loaded = load_complete_phase7_run(bundle / "model-runs" / "phase7")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent)).resolve()
        write_score_musicxml(
            loaded.validated_script,
            loaded.plan,
            loaded.score,
            temporary / "score.musicxml",
        )
        write_rendered_performance_smf(loaded.rendered, temporary / "final.mid")
        check_score_musicxml(
            loaded.validated_script,
            loaded.plan,
            loaded.score,
            temporary / "score.musicxml",
        )
        check_rendered_performance_smf(loaded.rendered, temporary / "final.mid")
        for filename in ("score.musicxml", "final.mid"):
            if sha256_file(temporary / filename) != sha256_file(bundle / "artifacts" / filename):
                raise ValueError(f"replayed artifact differs: {filename}")
        os.replace(temporary, target)
        temporary = None
        return Phase8ReplayResult(True, target, ())
    except (KeyError, OSError, TypeError, ValueError) as error:
        return _replay_failure(IssueCode.LINEAGE_MISMATCH, str(error))
    finally:
        if "temporary" in locals() and temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _verify_bundle(bundle: Path) -> tuple[ValidationIssue, ...]:
    try:
        manifest = _read_object(bundle / "manifest.json")
        schema_version = manifest.get("schema_version")
        if (
            manifest.get("bundle_type") != "scoim-generation-trial"
            or schema_version not in {1, 2}
            or manifest.get("target_profile") != "solo_piano_3m_v2"
        ):
            raise ValueError("bundle identity is invalid")
        files = cast(Mapping[str, object], manifest["files"])
        actual_paths = {
            path.relative_to(bundle).as_posix()
            for path in bundle.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        if set(files) != actual_paths:
            raise ValueError("bundle file inventory does not match")
        for relative_path, expected_hash in files.items():
            if sha256_file(bundle / relative_path) != expected_hash:
                raise ValueError(f"bundle file hash differs: {relative_path}")
        loaded = load_complete_phase7_run(bundle / "model-runs" / "phase7")
        for phase_name in _PHASE_NAMES:
            operation_records = check_finite_model_operation_records(
                bundle / "model-runs" / phase_name
            )
            if not operation_records.valid:
                raise ValueError(
                    f"{phase_name} model operation record is invalid: "
                    f"{operation_records.issues[0].message}"
                )
        expected_hashes = {
            "validated_script_sha256": sha256_json(loaded.validated_script),
            "piece_plan_sha256": sha256_json(asdict(loaded.plan)),
            "score_spec_sha256": sha256_json(asdict(loaded.score)),
            "performance_spec_sha256": sha256_json(asdict(loaded.performance)),
            "rendered_performance_sha256": sha256_json(asdict(loaded.rendered)),
        }
        if any(manifest.get(name) != digest for name, digest in expected_hashes.items()):
            raise ValueError("bundle lineage hash does not match")
        if schema_version == 2:
            composition_id = manifest.get("composition_id")
            trial_id = manifest.get("trial_id")
            composition_manifest_bytes = (
                bundle / "lineage" / "composition-manifest.json"
            ).read_bytes()
            composition_manifest = json.loads(composition_manifest_bytes.decode("utf-8"))
            if (
                not isinstance(composition_id, str)
                or not composition_id
                or not isinstance(trial_id, str)
                or not trial_id
                or not isinstance(composition_manifest, dict)
                or composition_manifest.get("bundle_type") != "composition"
                or composition_manifest.get("schema_version") != 3
                or composition_manifest.get("target_profile") != "solo_piano_3m_v2"
                or composition_manifest.get("composition_id") != composition_id
                or manifest.get("composition_manifest_sha256")
                != sha256_bytes(composition_manifest_bytes)
            ):
                raise ValueError("composition manifest lineage does not match")
        checks = _read_object(bundle / "checks.json")
        if any(
            cast(Mapping[str, object], value).get("status") != "passed" for value in checks.values()
        ):
            raise ValueError("bundle artifact check did not pass")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return (ValidationIssue(IssueCode.LINEAGE_MISMATCH, str(error), "/bundle"),)
    return ()


def _bundle_failure(code: IssueCode, message: str, target: Path) -> Phase8BundleResult:
    return Phase8BundleResult(
        False,
        None,
        (ValidationIssue(code, message, str(target)),),
    )


def _replay_failure(code: IssueCode, message: str) -> Phase8ReplayResult:
    return Phase8ReplayResult(
        False,
        None,
        (ValidationIssue(code, message, "/bundle"),),
    )


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return cast(dict[str, object], value)
