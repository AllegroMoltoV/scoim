import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scoim import realize
from scoim.phase8_bundle import verify_phase8_bundle

_FIXTURE = Path(__file__).parent / "fixtures/scoim/legacy-context-733a97e"


def test_pre_context_change_bundle_verifies_and_replays_without_a_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = json.loads((_FIXTURE / "provenance.json").read_text(encoding="utf-8"))
    archive_path = _FIXTURE / provenance["archive_file"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == provenance["archive_sha256"]
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(tmp_path / "bundle")
    assert (
        hashlib.sha256((tmp_path / "bundle/manifest.json").read_bytes()).hexdigest()
        == provenance["fixture_manifest_sha256"]
    )
    monkeypatch.chdir(tmp_path)

    verified = verify_phase8_bundle(Path("bundle"))
    assert verified.valid, verified.issues
    result = realize(Path("bundle"), Path("replay"))

    assert result.succeeded, result.issues
    assert result.input_kind == "trial"
    assert result.artifacts == {
        "final_musicxml": "artifacts/score.musicxml",
        "final_smf": "artifacts/final.mid",
    }
    for relative_path, expected_sha256 in provenance["artifact_sha256"].items():
        replayed = (Path("replay") / relative_path).read_bytes()
        assert replayed == (Path("bundle") / relative_path).read_bytes()
        assert hashlib.sha256(replayed).hexdigest() == expected_sha256
