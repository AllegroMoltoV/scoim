import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scoim import realize

_FIXTURE = Path(__file__).parent / "fixtures/scoim/legacy-timing-40d894f"


def test_pre_timing_change_bundle_replays_through_public_api_without_a_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = json.loads((_FIXTURE / "provenance.json").read_text(encoding="utf-8"))
    archive_path = _FIXTURE / provenance["archive_file"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == provenance["archive_sha256"]
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(tmp_path / "bundle")
    assert (
        hashlib.sha256((tmp_path / "bundle/manifest.json").read_bytes()).hexdigest()
        == (provenance["fixture_manifest_sha256"])
    )
    monkeypatch.chdir(tmp_path)

    result = realize(Path("bundle"), Path("replay"))

    assert result.succeeded, result.issues
    assert result.input_kind == "trial"
    assert result.artifacts == {
        "final_musicxml": "artifacts/score.musicxml",
        "final_smf": "artifacts/final.mid",
    }
    for relative_path, expected_sha256 in provenance["artifact_sha256"].items():
        assert hashlib.sha256((Path("replay") / relative_path).read_bytes()).hexdigest() == (
            expected_sha256
        )
