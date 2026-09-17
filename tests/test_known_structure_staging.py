from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_musical_composer.known_structure_staging import write_known_fixture_staging
from llm_musical_composer.run_state import sha256_file


def test_known_fixture_staging_renames_final_smf_and_preserves_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fixtures"
    fixture_dir = source / "fixture-a"
    fixture_dir.mkdir(parents=True)
    smf = fixture_dir / "final.mid"
    smf.write_bytes(b"MThd-fixture-a")
    records = tmp_path / "fixtures.jsonl"
    records.write_text(
        json.dumps(
            {
                "fixture_id": "fixture-a",
                "hashes": {"smf": sha256_file(smf)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fixture_manifest = tmp_path / "fixture-manifest.json"
    fixture_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "outputs": {"fixtures.jsonl": sha256_file(records)},
            }
        ),
        encoding="utf-8",
    )
    method_manifest = tmp_path / "method-manifest.json"
    method_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "llm_musical_composer.known_structure_staging.verify_method_manifest",
        lambda **_arguments: "f" * 64,
    )

    result = write_known_fixture_staging(
        fixtures_root=source,
        fixtures_manifest=fixture_manifest,
        fixture_records=records,
        repository_root=Path.cwd(),
        method_manifest=method_manifest,
        output_dir=tmp_path / "output",
    )

    assert result == {"status": "pass", "known_fixture_count": 1}
    manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
    assert manifest["split_role"] == "known_fixture"
    assert set(manifest["staged_smf"]) == {"fixture-a.mid"}
    assert (tmp_path / "output" / "staged-smf" / "fixture-a.mid").read_bytes() == smf.read_bytes()
