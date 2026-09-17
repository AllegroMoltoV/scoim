from __future__ import annotations

from pathlib import Path

from llm_musical_composer.harmonic_skeleton_run import run_harmonic_skeleton_diagnostics


def test_known_source_roundtrip_is_exact_and_deterministic(tmp_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[1]
    first = tmp_path / "first"
    second = tmp_path / "second"

    result = run_harmonic_skeleton_diagnostics(workspace, first)
    repeated = run_harmonic_skeleton_diagnostics(workspace, second)

    assert result == repeated
    assert result["status"] == "pass"
    assert result["assessed_case_count"] == 2
    assert result["passed_case_count"] == 2
    assert result["passed_family_count"] == 2
    names = sorted(path.name for path in first.iterdir())
    assert names == sorted(path.name for path in second.iterdir())
    assert len(names) == 7
    assert all((first / name).read_bytes() == (second / name).read_bytes() for name in names)
