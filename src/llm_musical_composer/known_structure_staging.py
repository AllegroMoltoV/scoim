"""既知構造fixtureのSMFを観測専用stagingへ隔離する。"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_musical_composer.observed_structure_method import verify_method_manifest
from llm_musical_composer.run_state import atomic_write_json, sha256_file


class KnownStructureStagingError(ValueError):
    """既知構造fixtureの入力またはstaging契約に違反した。"""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise KnownStructureStagingError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not all(isinstance(value, dict) for value in values):
        raise KnownStructureStagingError(f"JSONL records must be objects: {path}")
    return values


def write_known_fixture_staging(
    *,
    fixtures_root: Path,
    fixtures_manifest: Path,
    fixture_records: Path,
    repository_root: Path,
    method_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """fixtureのoracle注釈を渡さず、検証済みSMFだけを複製する。"""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise KnownStructureStagingError(f"output directory is not empty: {output_dir}")
    method_hash = verify_method_manifest(
        repository_root=Path(repository_root),
        manifest_path=Path(method_manifest),
    )
    fixture_manifest_payload = _read_json(Path(fixtures_manifest))
    outputs = fixture_manifest_payload.get("outputs")
    if (
        fixture_manifest_payload.get("schema_version") != 1
        or fixture_manifest_payload.get("status") not in {"pass", "passed"}
        or not isinstance(outputs, Mapping)
        or outputs.get("fixtures.jsonl") != sha256_file(Path(fixture_records))
    ):
        raise KnownStructureStagingError("fixture manifest contract is invalid")
    records = _read_jsonl(Path(fixture_records))
    staged: dict[str, str] = {}
    sources: list[tuple[Path, str]] = []
    for record in records:
        fixture_id = record.get("fixture_id")
        hashes = record.get("hashes")
        expected_hash = hashes.get("smf") if isinstance(hashes, Mapping) else None
        if (
            not isinstance(fixture_id, str)
            or Path(fixture_id).name != fixture_id
            or not isinstance(expected_hash, str)
        ):
            raise KnownStructureStagingError("fixture record contract is invalid")
        source = Path(fixtures_root) / fixture_id / "final.mid"
        if not source.is_file() or sha256_file(source) != expected_hash.lower():
            raise KnownStructureStagingError(f"fixture SMF SHA-256 mismatch: {fixture_id}")
        name = f"{fixture_id}.mid"
        if name in staged:
            raise KnownStructureStagingError(f"duplicate fixture staging name: {name}")
        staged[name] = expected_hash.lower()
        sources.append((source, name))

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / "staged-smf"
    staging_dir.mkdir()
    for source, name in sources:
        shutil.copy2(source, staging_dir / name)
    run_spec = {
        "schema_version": 2,
        "split_role": "known_fixture",
        "inputs": {
            "fixtures_manifest": sha256_file(Path(fixtures_manifest)),
            "fixture_records": sha256_file(Path(fixture_records)),
            "method_manifest": method_hash,
        },
        "renaming_rule": "<fixture_id>/final.mid -> <fixture_id>.mid",
    }
    atomic_write_json(output_dir / "run-spec.json", run_spec)
    manifest = {
        "schema_version": 2,
        "status": "pass",
        "split_role": "known_fixture",
        "inputs": {"method_manifest": method_hash},
        "outputs": {"run-spec.json": sha256_file(output_dir / "run-spec.json")},
        "staged_smf": dict(sorted(staged.items())),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return {"status": "pass", "known_fixture_count": len(staged)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-root", type=Path, required=True)
    parser.add_argument("--fixtures-manifest", type=Path, required=True)
    parser.add_argument("--fixture-records", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--method-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = write_known_fixture_staging(
        fixtures_root=arguments.fixtures_root,
        fixtures_manifest=arguments.fixtures_manifest,
        fixture_records=arguments.fixture_records,
        repository_root=arguments.repository_root,
        method_manifest=arguments.method_manifest,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
