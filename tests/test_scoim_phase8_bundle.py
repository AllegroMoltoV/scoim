import json
import zipfile
from pathlib import Path

import pytest
from test_scoim_phase7_state import _phase7_run, _rewrite_as_phase7_schema_v1

from llm_musical_composer.run_state import sha256_bytes, sha256_file, sha256_json
from scoim.phase7_state import load_complete_phase7_run
from scoim.phase8_bundle import (
    Phase8BundleRequest,
    create_phase8_bundle,
    replay_phase8_bundle,
    verify_phase8_bundle,
)
from scoim.score_rendering import check_rendered_performance_smf, write_rendered_performance_smf


def _rewrite_bundle_phase7_as_v1(bundle_dir: Path, *, bundle_schema_version: int) -> None:
    embedded_phase7 = bundle_dir / "model-runs" / "phase7"
    _rewrite_as_phase7_schema_v1(embedded_phase7)
    legacy_rendered = json.loads(
        (embedded_phase7 / "outputs" / "rendered-performance.json").read_text(encoding="utf-8")
    )
    input_rendered_path = bundle_dir / "inputs" / "rendered-performance.json"
    input_rendered_path.write_text(json.dumps(legacy_rendered), encoding="utf-8")
    loaded = load_complete_phase7_run(embedded_phase7)
    smf_path = bundle_dir / "artifacts" / "final.mid"
    write_rendered_performance_smf(loaded.rendered, smf_path)
    checks_path = bundle_dir / "checks.json"
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    checks["smf"] = check_rendered_performance_smf(loaded.rendered, smf_path)
    assert checks["smf"]["status"] == "passed"
    checks_path.write_text(json.dumps(checks), encoding="utf-8")
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = bundle_schema_version
    manifest["rendered_performance_sha256"] = sha256_json(legacy_rendered)
    for relative_path in (
        "artifacts/final.mid",
        "checks.json",
        "inputs/rendered-performance.json",
        "model-runs/phase7/run-spec.json",
        "model-runs/phase7/outputs/phase7-state.json",
        "model-runs/phase7/outputs/rendered-performance.json",
    ):
        manifest["files"][relative_path] = sha256_file(bundle_dir / relative_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


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
                "score": tmp_path / "score",
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
    assert manifest["schema_version"] == 5
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


@pytest.mark.parametrize(
    "fixture", ["legacy-timing-40d894f", "legacy-context-733a97e", "legacy-section-2e56cd7"]
)
def test_genuine_legacy_bundles_verify_and_replay(tmp_path, fixture):
    archive = Path(__file__).parent / "fixtures/scoim" / fixture / "bundle.zip"
    bundle = tmp_path / "bundle"
    with zipfile.ZipFile(archive) as saved:
        saved.extractall(bundle)
    verified = verify_phase8_bundle(bundle)
    assert verified.valid, verified.issues
    replayed = replay_phase8_bundle(bundle, tmp_path / "replay")
    assert replayed.replayed, replayed.issues
    for name in ("score.musicxml", "final.mid"):
        assert sha256_file(tmp_path / "replay" / name) == sha256_file(bundle / "artifacts" / name)


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_bundle_versions_keep_their_saved_phase7_v1_contract(tmp_path, version):
    archive = Path(__file__).parent / "fixtures/scoim/legacy-timing-40d894f/bundle.zip"
    bundle = tmp_path / "bundle"
    with zipfile.ZipFile(archive) as saved:
        saved.extractall(bundle)
    _rewrite_bundle_phase7_as_v1(bundle, bundle_schema_version=version)
    verified = verify_phase8_bundle(bundle)
    assert verified.valid, verified.issues
    replayed = replay_phase8_bundle(bundle, tmp_path / "replay")
    assert replayed.replayed, replayed.issues
    for name in ("score.musicxml", "final.mid"):
        assert sha256_file(tmp_path / "replay" / name) == sha256_file(bundle / "artifacts" / name)


def test_phase8_schema_v3_bundle_rejects_an_embedded_phase7_v1_contract(
    tmp_path: Path,
) -> None:
    phase7_dir = _phase7_run(tmp_path)
    bundle_dir = tmp_path / "bundle"
    composition_manifest = json.dumps(
        {
            "bundle_type": "composition",
            "schema_version": 3,
            "target_profile": "solo_piano_3m_v2",
            "composition_id": "composition-001",
        },
        sort_keys=True,
    ).encode()
    created = create_phase8_bundle(
        Phase8BundleRequest(
            phase7_run_dir=phase7_dir,
            phase_run_dirs={
                "phase3": tmp_path / "phase3",
                "score": tmp_path / "score",
            },
            composition_id="composition-001",
            trial_id="trial-invalid",
            composition_manifest=composition_manifest,
        ),
        bundle_dir,
    )
    assert created.created is True, created.issues
    _rewrite_bundle_phase7_as_v1(bundle_dir, bundle_schema_version=3)

    verified = verify_phase8_bundle(bundle_dir)

    assert verified.valid is False
    assert "phase-7 schema version 2" in verified.issues[0].message


def test_phase8_schema_v4_bundle_rejects_a_phase7_v1_contract(tmp_path: Path) -> None:
    phase7_dir = _phase7_run(tmp_path)
    _rewrite_as_phase7_schema_v1(phase7_dir)

    result = create_phase8_bundle(
        Phase8BundleRequest(
            phase7_run_dir=phase7_dir,
            phase_run_dirs={
                "phase3": tmp_path / "phase3",
                "score": tmp_path / "score",
            },
            composition_id="composition-001",
            trial_id="trial-legacy",
            composition_manifest=json.dumps(
                {
                    "bundle_type": "composition",
                    "schema_version": 3,
                    "target_profile": "solo_piano_3m_v2",
                    "composition_id": "composition-001",
                }
            ).encode(),
        ),
        tmp_path / "bundle",
    )

    assert result.created is False
    assert result.issues[0].code.value == "lineage_mismatch"
    assert "phase-7 schema version 3" in result.issues[0].message


def test_phase8_bundle_rejects_a_non_string_lineage_identifier(tmp_path: Path) -> None:
    phase7_dir = _phase7_run(tmp_path)

    result = create_phase8_bundle(
        Phase8BundleRequest(
            phase7_run_dir=phase7_dir,
            phase_run_dirs={
                "phase3": tmp_path / "phase3",
                "score": tmp_path / "score",
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
    phase4_dir = tmp_path / "score"
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
                "score": phase4_dir,
            },
            composition_id="composition-001",
            trial_id="trial-001",
            composition_manifest=composition_manifest,
        ),
        tmp_path / "bundle",
    )

    assert result.created is False
    assert result.issues[0].code.value == "lineage_mismatch"
    assert "score" in result.issues[0].message


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
                "score": tmp_path / "score",
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


def test_v5_rejects_missing_score_boundary_even_with_updated_file_inventory(tmp_path):
    phase7 = _phase7_run(tmp_path)
    bundle = tmp_path / "bundle"
    created = create_phase8_bundle(
        Phase8BundleRequest(
            phase7_run_dir=phase7,
            phase_run_dirs={"phase3": tmp_path / "phase3", "score": tmp_path / "score"},
            composition_id="composition-001",
            trial_id="trial-001",
            composition_manifest=json.dumps(
                {
                    "bundle_type": "composition",
                    "schema_version": 3,
                    "target_profile": "solo_piano_3m_v2",
                    "composition_id": "composition-001",
                }
            ).encode(),
        ),
        bundle,
    )
    assert created.created, created.issues
    relative = "model-runs/score/outputs/score-state.json"
    (bundle / relative).unlink()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["files"][relative]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    verified = verify_phase8_bundle(bundle)
    assert not verified.valid
    assert "score-state.json" in verified.issues[0].message
