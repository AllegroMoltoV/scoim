from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_musical_composer.run_state import sha256_file
from llm_musical_composer.texture_edge_diagnostic_run import (
    TextureEdgeDiagnosticRunError,
    _input_processing_status,
    development_observation_sources,
)


def _write_development_split(root: Path) -> tuple[Path, Path]:
    staging = root / "development-smf"
    staging.mkdir(parents=True)
    (staging / "alpha.mid").write_bytes(b"alpha")
    (staging / "beta.mid").write_bytes(b"beta")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "development_smf": {
                    path.name: sha256_file(path) for path in sorted(staging.iterdir())
                },
            }
        ),
        encoding="utf-8",
    )
    return staging, manifest


def test_development_sources_are_manifest_verified_and_deterministically_ordered(
    tmp_path: Path,
) -> None:
    staging, manifest = _write_development_split(tmp_path)

    sources = development_observation_sources(
        staging_dir=staging,
        staging_manifest=manifest,
    )

    assert [source.case_id for source in sources] == ["development-alpha", "development-beta"]
    assert all(source.group_id == "development-reference-unlabeled" for source in sources)


def test_development_sources_reject_a_changed_staged_file(tmp_path: Path) -> None:
    staging, manifest = _write_development_split(tmp_path)
    (staging / "alpha.mid").write_bytes(b"changed")

    with pytest.raises(TextureEdgeDiagnosticRunError, match="SHA-256 mismatch"):
        development_observation_sources(
            staging_dir=staging,
            staging_manifest=manifest,
        )


def test_input_processing_status_uses_the_supplied_source_counts() -> None:
    texture_records = [{"status": "assessed"} for _ in range(3)]
    voice_records = [{"status": "assessed"} for _ in range(2)]

    assert (
        _input_processing_status(
            texture_records,
            voice_records,
            expected_source_count=3,
            expected_known_voice_source_count=2,
        )
        == "pass"
    )
    assert (
        _input_processing_status(
            texture_records,
            voice_records,
            expected_source_count=4,
            expected_known_voice_source_count=2,
        )
        == "partial"
    )
