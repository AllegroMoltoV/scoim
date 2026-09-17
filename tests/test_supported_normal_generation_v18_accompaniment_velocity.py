import json
from pathlib import Path

import pytest

import llm_musical_composer.supported_normal_generation_v18_accompaniment_velocity as v18_module
from llm_musical_composer.supported_normal_generation_v18_accompaniment_velocity import (
    SupportedNormalGenerationV18Error,
    run_supported_normal_generation_v18,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_v18_changes_only_velocity_and_improves_accompaniment_repetition(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "v18"

    result = run_supported_normal_generation_v18(PROJECT_ROOT, artifact_root)

    assert result["status"] == "completed_unfit"
    assert result["confirmed_external_call_count"] == 0
    assert result["generic_quality_passes"] == {
        "score": True,
        "pipeline": True,
        "rendered_texture_budget": True,
    }
    assert result["invariants"]["velocity_only_change"] is True
    assert result["velocity_diagnostic_comparison"]["policy_applied"] is True
    assert result["velocity_diagnostic_comparison"]["improved"] is True
    assert result["key_release_calibration"]["selected_percent"] == 59
    assert result["promoted"] is False
    smf = artifact_root / "runs/whole-score/staged/final.mid"
    assert smf.is_file()
    diagnostic = json.loads(
        (
            artifact_root
            / "runs/whole-score/outputs/voice-velocity-diagnostic.json"
        ).read_text(encoding="utf-8")
    )
    assert diagnostic["candidate"]["velocity_policy_id"] == (
        "foreground-accompaniment-harmony-shape-v1"
    )


def test_v18_rejects_changed_v17_performance_before_creating_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original = v18_module.sha256_file

    def tampered(path: Path) -> str:
        if path.name == "performance-spec.dsl":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(v18_module, "sha256_file", tampered)

    with pytest.raises(SupportedNormalGenerationV18Error, match="SHA-256"):
        run_supported_normal_generation_v18(PROJECT_ROOT, tmp_path / "v18")

    assert not (tmp_path / "v18").exists()
