import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    ("scripts/copy-reference-corpus.ps1", "scripts/copy-source-smf.ps1"),
)
def test_corpus_copy_scripts_require_the_source_location_from_the_caller(
    relative_path: str,
) -> None:
    script = (ROOT / relative_path).read_text(encoding="utf-8")

    assert re.search(r"\\\\\d{1,3}(?:\.\d{1,3}){3}\\", script) is None
    assert "[Parameter(Mandatory = $true)][string]$SourceRoot" in script


def test_package_check_installs_the_version_read_from_distribution_metadata() -> None:
    package_test = (ROOT / "scripts/test-package.ps1").read_text(encoding="utf-8")
    distribution_check = (ROOT / "scripts/check-distributions.py").read_text(encoding="utf-8")

    assert "scoim==0.1.0" not in package_test
    assert "--version-file" in package_test
    assert "--version-file" in distribution_check


def test_public_package_version_has_a_changelog_entry() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert f"## [{project['project']['version']}]" in changelog


def test_public_music_omits_audio_with_unproven_renderer_rights() -> None:
    example_root = ROOT / "examples/public-music/bright-return"

    assert not tuple(example_root.glob("*/preview.mp3"))
    for path in (
        example_root / "README.md",
        example_root / "take-01/provenance.json",
        example_root / "take-02/provenance.json",
    ):
        assert "preview.mp3" not in path.read_text(encoding="utf-8")


def test_public_clone_ignores_documented_local_workspaces() -> None:
    ignored = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())

    assert {
        ".appendix/",
        ".beads/",
        ".dolt/",
        ".logs/",
        ".prompts/",
        ".tmp/",
        "AGENTS.md",
        "docs/reports/",
    } <= ignored
