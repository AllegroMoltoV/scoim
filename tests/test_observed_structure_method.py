from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from llm_musical_composer.observed_structure_method import (
    METHOD_CONSTANTS,
    ObservedStructureMethodError,
    current_git_commit,
    verify_method_manifest,
)
from llm_musical_composer.run_state import sha256_file


def _repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _manifest(tmp_path: Path, *, file_hash: str | None = None) -> Path:
    root = _repository_root()
    constants_payload = json.dumps(
        METHOD_CONSTANTS,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    import hashlib

    path = tmp_path / "method-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "frozen",
                "frozen_commit": current_git_commit(root),
                "files": {
                    "pyproject.toml": file_hash or sha256_file(root / "pyproject.toml")
                },
                "constants": METHOD_CONSTANTS,
                "constants_sha256": hashlib.sha256(constants_payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_verifier_accepts_matching_commit_file_and_constants(tmp_path: Path) -> None:
    root = _repository_root()
    manifest = _manifest(tmp_path)

    assert verify_method_manifest(
        repository_root=root,
        manifest_path=manifest,
        expected_files=("pyproject.toml",),
    ) == sha256_file(manifest)


def test_verifier_rejects_file_hash_mismatch(tmp_path: Path) -> None:
    root = _repository_root()
    manifest = _manifest(tmp_path, file_hash="0" * 64)

    with pytest.raises(ObservedStructureMethodError, match="file SHA-256 mismatch"):
        verify_method_manifest(
            repository_root=root,
            manifest_path=manifest,
            expected_files=("pyproject.toml",),
        )
