"""観測候補V2の凍結方法を作成・検証する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_musical_composer.run_state import atomic_write_json, sha256_file

METHOD_CONSTANTS: dict[str, Any] = {
    "holdout_count": 8,
    "score_timing": {
        "grouping_profiles": ["attack-30ms", "rolled-merge-60ms", "rolled-separate"],
        "score_grids": ["binary", "binary-dotted", "binary-dotted-triplet"],
        "time_map_segment_counts": [1, 4, 8],
        "candidate_count_per_source": 27,
    },
    "observed_structure": {
        "boundary_window_sizes": [4, 8, 16],
        "recurrence_window_sizes": [3, 4, 5, 6, 7, 8],
        "recurrence_extension_probe_size": 9,
        "maximum_boundary_selection_rate": 0.5,
        "minimum_assessed_source_count_per_candidate_kind": 2,
    },
}

FROZEN_METHOD_FILES = (
    "docs/design/reference-smf-reverse-decomposition.md",
    "pyproject.toml",
    "src/llm_musical_composer/known_structure_evaluation.py",
    "src/llm_musical_composer/known_structure_staging.py",
    "src/llm_musical_composer/observed_structure_candidates.py",
    "src/llm_musical_composer/observed_structure_method.py",
    "src/llm_musical_composer/reference_decomposition.py",
    "src/llm_musical_composer/reference_profile.py",
    "src/llm_musical_composer/score_timing_development_run.py",
    "src/llm_musical_composer/score_timing_holdout_split.py",
    "src/llm_musical_composer/score_timing_hypothesis.py",
    "src/llm_musical_composer/score_timing_split.py",
    "src/llm_musical_composer/score_timing_upper_evidence.py",
    "tests/test_known_structure_evaluation.py",
    "tests/test_known_structure_staging.py",
    "tests/test_observed_structure_candidates.py",
    "tests/test_observed_structure_method.py",
    "tests/test_score_timing_development_run.py",
    "tests/test_score_timing_split.py",
    "tests/test_score_timing_upper_evidence.py",
)


class ObservedStructureMethodError(ValueError):
    """凍結方法の宣言と現在の実装が一致しない。"""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ObservedStructureMethodError(
            f"unable to read method manifest: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ObservedStructureMethodError("method manifest root must be an object")
    return value


def _git(repository_root: Path, *arguments: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=text,
    )
    if result.returncode:
        stderr = result.stderr.strip() if text else result.stderr.decode(errors="replace").strip()
        raise ObservedStructureMethodError(
            f"git {' '.join(arguments)} failed: {stderr}"
        )
    return result.stdout.strip() if text else result.stdout


def current_git_commit(repository_root: Path) -> str:
    """現在のHEAD commitを返す。"""
    return str(_git(Path(repository_root), "rev-parse", "HEAD"))


def _validated_relative_paths(paths: Sequence[str]) -> tuple[str, ...]:
    result = []
    for value in paths:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
            raise ObservedStructureMethodError(f"method file path is invalid: {value}")
        result.append(value)
    if len(set(result)) != len(result):
        raise ObservedStructureMethodError("method file paths must be unique")
    return tuple(result)


def write_method_manifest(
    *,
    repository_root: Path,
    output_path: Path,
    method_files: Sequence[str] = FROZEN_METHOD_FILES,
) -> dict[str, Any]:
    """HEADと一致する方法ファイルだけを凍結manifestへ保存する。"""
    repository_root = Path(repository_root).resolve()
    files: dict[str, str] = {}
    for relative in _validated_relative_paths(method_files):
        path = repository_root / relative
        if not path.is_file():
            raise ObservedStructureMethodError(f"method file is missing: {relative}")
        current_hash = sha256_file(path)
        committed = _git(repository_root, "show", f"HEAD:{relative}", text=False)
        committed_hash = hashlib.sha256(committed).hexdigest()
        if current_hash != committed_hash:
            raise ObservedStructureMethodError(
                f"method file differs from HEAD: {relative}"
            )
        files[relative] = current_hash
    payload = {
        "schema_version": 1,
        "status": "frozen",
        "frozen_commit": current_git_commit(repository_root),
        "files": files,
        "constants": METHOD_CONSTANTS,
        "constants_sha256": _sha256_value(METHOD_CONSTANTS),
    }
    atomic_write_json(Path(output_path), payload)
    return payload


def verify_method_manifest(
    *,
    repository_root: Path,
    manifest_path: Path,
    expected_files: Sequence[str] = FROZEN_METHOD_FILES,
) -> str:
    """manifest、HEAD、方法ファイル、定数を再計算して照合する。"""
    repository_root = Path(repository_root).resolve()
    manifest_path = Path(manifest_path)
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "frozen"
        or not isinstance(files, Mapping)
    ):
        raise ObservedStructureMethodError("method manifest contract is invalid")
    if manifest.get("frozen_commit") != current_git_commit(repository_root):
        raise ObservedStructureMethodError("current HEAD differs from frozen commit")
    declared = {str(name): str(digest).lower() for name, digest in files.items()}
    expected = set(_validated_relative_paths(expected_files))
    if set(declared) != expected:
        raise ObservedStructureMethodError("method file set differs from expected files")
    for relative, expected_hash in declared.items():
        path = repository_root / relative
        if not path.is_file() or sha256_file(path).lower() != expected_hash:
            raise ObservedStructureMethodError(f"method file SHA-256 mismatch: {relative}")
        committed = _git(repository_root, "show", f"HEAD:{relative}", text=False)
        if hashlib.sha256(committed).hexdigest() != expected_hash:
            raise ObservedStructureMethodError(
                f"frozen commit file SHA-256 mismatch: {relative}"
            )
    if manifest.get("constants") != METHOD_CONSTANTS:
        raise ObservedStructureMethodError("method constants differ from current constants")
    if manifest.get("constants_sha256") != _sha256_value(METHOD_CONSTANTS):
        raise ObservedStructureMethodError("method constants SHA-256 mismatch")
    return sha256_file(manifest_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = write_method_manifest(
        repository_root=arguments.repository_root,
        output_path=arguments.output,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
