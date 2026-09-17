import json
from pathlib import Path

from test_scoim_phase7_state import _phase7_run

from llm_musical_composer.run_state import sha256_bytes, sha256_file
from scoim.phase8_bundle import (
    Phase8BundleRequest,
    create_phase8_bundle,
    replay_phase8_bundle,
    verify_phase8_bundle,
)


def test_phase8_bundle_replays_final_artifacts_without_model_access(tmp_path: Path) -> None:
    phase7_dir = _phase7_run(tmp_path)
    bundle_dir = tmp_path / "bundle"
    composition_manifest = (
        json.dumps(
            {
                "bundle_type": "composition",
                "schema_version": 3,
                "target_profile": "solo_piano_3m_v2",
                "composition_id": "composition-001",
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()

    created = create_phase8_bundle(
        Phase8BundleRequest(
            phase7_run_dir=phase7_dir,
            phase_run_dirs={
                "phase3": tmp_path / "phase3",
                "phase4": tmp_path / "phase4",
                "phase5": tmp_path / "phase5",
                "phase6": tmp_path / "phase6",
            },
            composition_id="composition-001",
            trial_id="trial-001",
            composition_manifest=composition_manifest,
        ),
        bundle_dir,
    )

    assert created.created is True
    assert (bundle_dir / "artifacts" / "score.musicxml").is_file()
    assert (bundle_dir / "artifacts" / "final.mid").is_file()
    assert (bundle_dir / "lineage" / "composition-manifest.json").read_bytes() == (
        composition_manifest
    )
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["composition_id"] == "composition-001"
    assert manifest["trial_id"] == "trial-001"
    assert manifest["composition_manifest_sha256"] == sha256_bytes(composition_manifest)

    replay_dir = tmp_path / "replay"
    replayed = replay_phase8_bundle(bundle_dir, replay_dir)

    assert replayed.replayed is True
    assert sha256_file(replay_dir / "score.musicxml") == sha256_file(
        bundle_dir / "artifacts" / "score.musicxml"
    )
    assert sha256_file(replay_dir / "final.mid") == sha256_file(
        bundle_dir / "artifacts" / "final.mid"
    )


def test_phase8_bundle_rejects_a_non_string_lineage_identifier(tmp_path: Path) -> None:
    phase7_dir = _phase7_run(tmp_path)

    result = create_phase8_bundle(
        Phase8BundleRequest(
            phase7_run_dir=phase7_dir,
            phase_run_dirs={
                "phase3": tmp_path / "phase3",
                "phase4": tmp_path / "phase4",
                "phase5": tmp_path / "phase5",
                "phase6": tmp_path / "phase6",
            },
            composition_id=123,  # type: ignore[arg-type]
            trial_id="trial-001",
            composition_manifest=b"{}",
        ),
        tmp_path / "bundle",
    )

    assert result.created is False
    assert result.issues[0].code.value == "lineage_mismatch"


def test_phase8_bundle_rejects_an_unverifiable_phase_operation_record(
    tmp_path: Path,
) -> None:
    phase7_dir = _phase7_run(tmp_path)
    phase4_dir = tmp_path / "phase4"
    validation_path = next((phase4_dir / "attempts").glob("*/attempt-*/validation.json"))
    validation_path.unlink()
    composition_manifest = json.dumps(
        {
            "bundle_type": "composition",
            "schema_version": 3,
            "target_profile": "solo_piano_3m_v2",
            "composition_id": "composition-001",
        },
        sort_keys=True,
    ).encode()

    result = create_phase8_bundle(
        Phase8BundleRequest(
            phase7_run_dir=phase7_dir,
            phase_run_dirs={
                "phase3": tmp_path / "phase3",
                "phase4": phase4_dir,
                "phase5": tmp_path / "phase5",
                "phase6": tmp_path / "phase6",
            },
            composition_id="composition-001",
            trial_id="trial-001",
            composition_manifest=composition_manifest,
        ),
        tmp_path / "bundle",
    )

    assert result.created is False
    assert result.issues[0].code.value == "lineage_mismatch"
    assert "phase4" in result.issues[0].message


def test_phase8_bundle_rejects_a_modified_embedded_composition_manifest(
    tmp_path: Path,
) -> None:
    phase7_dir = _phase7_run(tmp_path)
    composition_manifest = json.dumps(
        {
            "bundle_type": "composition",
            "schema_version": 3,
            "target_profile": "solo_piano_3m_v2",
            "composition_id": "composition-001",
        },
        sort_keys=True,
    ).encode()
    bundle = tmp_path / "bundle"
    created = create_phase8_bundle(
        Phase8BundleRequest(
            phase7_run_dir=phase7_dir,
            phase_run_dirs={
                "phase3": tmp_path / "phase3",
                "phase4": tmp_path / "phase4",
                "phase5": tmp_path / "phase5",
                "phase6": tmp_path / "phase6",
            },
            composition_id="composition-001",
            trial_id="trial-001",
            composition_manifest=composition_manifest,
        ),
        bundle,
    )
    assert created.created is True, created.issues
    (bundle / "lineage" / "composition-manifest.json").write_bytes(b"{}")

    verified = verify_phase8_bundle(bundle)

    assert verified.valid is False
    assert verified.issues[0].code.value == "lineage_mismatch"
