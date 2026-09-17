from __future__ import annotations

from pathlib import Path

from llm_musical_composer.piano_texture_register_placement_run import (
    run_register_placement_diagnostics,
)


def test_saved_diagnostics_cover_two_source_families_and_are_deterministic(
    tmp_path: Path,
) -> None:
    workspace = Path(__file__).resolve().parents[1]
    first = tmp_path / "first"
    second = tmp_path / "second"

    result = run_register_placement_diagnostics(workspace, first)
    repeated = run_register_placement_diagnostics(workspace, second)

    assert result["status"] == "pass"
    assert result["direct_regression"]["passed_case_count"] == 1
    assert result["known_sources"]["assessed_case_count"] == 4
    assert result["known_sources"]["passed_family_count"] == 2
    assert result["fixtures"]["passed_case_count"] == 2
    assert repeated == result
    assert sorted(path.name for path in first.iterdir()) == sorted(
        path.name for path in second.iterdir()
    )
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
