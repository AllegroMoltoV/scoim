import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest
from test_scoim_phase7_state import _legacy_phase7_run, _phase7_run, _rewrite_as_phase7_schema_v1

from llm_musical_composer.run_state import sha256_bytes, sha256_file, sha256_json
from scoim.phase7_state import load_complete_phase7_run
from scoim.phase8_bundle import (
    Phase8BundleRequest,
    create_phase8_bundle,
    replay_phase8_bundle,
    verify_phase8_bundle,
)
from scoim.public_realization import realize
from scoim.score_rendering import check_rendered_performance_smf, write_rendered_performance_smf
from scoim.script_0_4_validation import script_0_4_content_sha256


def _composition_manifest(phase7: Path) -> bytes:
    script = json.loads((phase7 / "inputs/validated-script.json").read_text(encoding="utf-8"))
    return json.dumps(
        {
            "bundle_type": "composition",
            "schema_version": 3,
            "target_profile": "solo_piano_3m_v2",
            "composition_id": "composition-001",
            "validated_script_content_sha256": script_0_4_content_sha256(script),
        },
        sort_keys=True,
    ).encode()


@pytest.mark.parametrize("entry", ["direct", "public"])
@pytest.mark.parametrize("destination", ["same", "parent", "child", "relative", "alias"])
def test_replay_rejects_source_destinations_without_writing(
    tmp_path, monkeypatch, entry, destination
):
    archive = Path(__file__).parent / "fixtures/scoim/legacy-pedal-phrase_legato-480904e/bundle.zip"
    bundle = tmp_path / "bundle"
    with zipfile.ZipFile(archive) as saved:
        saved.extractall(bundle)
    if destination == "same":
        target = bundle
    elif destination == "parent":
        target = tmp_path
    elif destination == "child":
        target = bundle / "new-directory/replay"
    elif destination == "relative":
        monkeypatch.chdir(tmp_path)
        target = Path("bundle/../bundle/new-directory/replay")
    else:
        alias = tmp_path / "bundle-alias"
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(alias), str(bundle)],
                check=True,
                capture_output=True,
            )
            assert alias.is_junction()
        else:
            alias.symlink_to(bundle, target_is_directory=True)
        assert alias.resolve() == bundle.resolve()
        target = alias / "new-directory/replay"

    def snapshot():
        return {
            path.relative_to(bundle).as_posix(): sha256_file(path) if path.is_file() else None
            for path in bundle.rglob("*")
        }

    before = snapshot()
    if entry == "public":
        result = realize(bundle, target)
        assert not result.succeeded
        assert not result.persisted
        assert result.artifacts == {}
    else:
        result = replay_phase8_bundle(bundle, target)
        assert not result.replayed
        assert result.output_dir is None
    assert result.issues[0].code.value == "storage_conflict"
    assert snapshot() == before
    checked = verify_phase8_bundle(bundle)
    assert checked.valid, checked.issues


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
    composition_manifest = _composition_manifest(phase7_dir)

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
    assert manifest["schema_version"] == 6
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
    "fixture",
    [
        "legacy-timing-40d894f",
        "legacy-context-733a97e",
        "legacy-section-2e56cd7",
        "legacy-pedal-phrase_legato-480904e",
        "legacy-pedal-harmony_legato-480904e",
    ],
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


def _refresh_bundle_inventory(bundle: Path) -> None:
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"] = {
        item.relative_to(bundle).as_posix(): sha256_file(item)
        for item in bundle.rglob("*")
        if item.is_file() and item.name != "manifest.json"
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("corruption", ["input", "state", "missing", "failed", "accepted"])
def test_legacy_bundle_rejects_broken_phase_lineage(tmp_path, corruption):
    bundle = tmp_path / "bundle"
    archive = Path(__file__).parent / "fixtures/scoim/legacy-timing-40d894f/bundle.zip"
    with zipfile.ZipFile(archive) as saved:
        saved.extractall(bundle)
    assert verify_phase8_bundle(bundle).valid
    if corruption == "input":
        path = bundle / "model-runs/phase6/inputs/phase5-state.json"
    elif corruption == "state":
        path = bundle / "model-runs/phase6/outputs/phase6-state.json"
    elif corruption == "missing":
        path = bundle / "model-runs/phase4/outputs/phase4-state.json"
    elif corruption == "accepted":
        path = next((bundle / "model-runs/phase3/events").glob("*/accepted.json"))
    else:
        path = next((bundle / "model-runs/phase4/attempts").glob("*/attempt-*/terminal.json"))
    if corruption in {"missing", "accepted"}:
        path.unlink()
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        value["status" if corruption == "failed" else "outcome"] = "failed"
        path.write_text(json.dumps(value), encoding="utf-8")
    _refresh_bundle_inventory(bundle)
    checked = verify_phase8_bundle(bundle)
    assert not checked.valid
    assert checked.issues[0].code.value == "lineage_mismatch"


def test_phase8_schema_v3_bundle_rejects_an_embedded_phase7_v1_contract(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    archive = Path(__file__).parent / "fixtures/scoim/legacy-timing-40d894f/bundle.zip"
    with zipfile.ZipFile(archive) as saved:
        saved.extractall(bundle_dir)
    _rewrite_bundle_phase7_as_v1(bundle_dir, bundle_schema_version=3)

    verified = verify_phase8_bundle(bundle_dir)

    assert verified.valid is False
    assert "phase-7 schema version 2" in verified.issues[0].message


def test_phase8_schema_v6_bundle_rejects_a_phase7_v1_contract(tmp_path: Path) -> None:
    phase7_dir = _legacy_phase7_run(tmp_path)
    _rewrite_as_phase7_schema_v1(phase7_dir)

    result = create_phase8_bundle(
        Phase8BundleRequest(
            phase7_run_dir=phase7_dir,
            phase_run_dirs={
                "phase3": phase7_dir.parent / "phase3",
                "score": phase7_dir.parent / "score",
            },
            composition_id="composition-001",
            trial_id="trial-legacy",
            composition_manifest=_composition_manifest(phase7_dir),
        ),
        tmp_path / "bundle",
    )

    assert result.created is False
    assert result.issues[0].code.value == "lineage_mismatch"
    assert "phase-7 schema version 4" in result.issues[0].message


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


@pytest.mark.parametrize("corruption", ["validation", "failed", "accepted"])
def test_phase8_bundle_rejects_an_unverifiable_phase_operation_record(
    tmp_path: Path,
    corruption: str,
) -> None:
    phase7_dir = _phase7_run(tmp_path)
    phase4_dir = tmp_path / "score"
    if corruption == "validation":
        next((phase4_dir / "attempts").glob("*/attempt-*/validation.json")).unlink()
    elif corruption == "accepted":
        next((phase4_dir / "events").glob("*/accepted.json")).unlink()
    else:
        path = next((phase4_dir / "attempts").glob("*/attempt-*/terminal.json"))
        value = json.loads(path.read_text(encoding="utf-8"))
        value["status"] = "failed"
        path.write_text(json.dumps(value), encoding="utf-8")
    composition_manifest = _composition_manifest(phase7_dir)

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


@pytest.mark.parametrize("corruption", ["missing", "different"])
def test_new_bundle_requires_the_composition_script_hash(tmp_path, corruption):
    phase7 = _phase7_run(tmp_path)
    manifest = json.loads(_composition_manifest(phase7))
    if corruption == "missing":
        del manifest["validated_script_content_sha256"]
    else:
        manifest["validated_script_content_sha256"] = "0" * 64
    target = tmp_path / "bundle"
    result = create_phase8_bundle(
        Phase8BundleRequest(
            phase7,
            {"phase3": tmp_path / "phase3", "score": tmp_path / "score"},
            "composition-001",
            "trial-001",
            json.dumps(manifest).encode(),
        ),
        target,
    )
    assert not result.created
    assert result.issues[0].code.value == "lineage_mismatch"
    assert "script hash" in result.issues[0].message
    assert not target.exists()


def test_legacy_bundle_rejects_a_mismatched_composition_script_hash(tmp_path):
    archive = Path(__file__).parent / "fixtures/scoim/legacy-timing-40d894f/bundle.zip"
    bundle = tmp_path / "bundle"
    with zipfile.ZipFile(archive) as saved:
        saved.extractall(bundle)
    path = bundle / "lineage/composition-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["validated_script_content_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle_manifest_path = bundle / "manifest.json"
    manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    manifest["composition_manifest_sha256"] = sha256_bytes(path.read_bytes())
    bundle_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_bundle_inventory(bundle)
    checked = verify_phase8_bundle(bundle)
    assert not checked.valid
    assert "script hash" in checked.issues[0].message


def test_phase8_bundle_rejects_a_modified_embedded_composition_manifest(
    tmp_path: Path,
) -> None:
    phase7_dir = _phase7_run(tmp_path)
    composition_manifest = _composition_manifest(phase7_dir)
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


def test_v6_rejects_missing_score_boundary_even_with_updated_file_inventory(tmp_path):
    phase7 = _phase7_run(tmp_path)
    bundle = tmp_path / "bundle"
    created = create_phase8_bundle(
        Phase8BundleRequest(
            phase7_run_dir=phase7,
            phase_run_dirs={"phase3": tmp_path / "phase3", "score": tmp_path / "score"},
            composition_id="composition-001",
            trial_id="trial-001",
            composition_manifest=_composition_manifest(phase7),
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
