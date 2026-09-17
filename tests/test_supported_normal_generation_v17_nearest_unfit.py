import json
from pathlib import Path

import pytest

import llm_musical_composer.supported_normal_generation_v17_nearest_unfit as v17_module
from llm_musical_composer.supported_normal_generation_v17_nearest_unfit import (
    SupportedNormalGenerationV17Error,
    run_supported_normal_generation_v17,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_v17_replays_v16_as_a_zero_call_unfit_candidate(tmp_path: Path) -> None:
    artifact_root = tmp_path / "v17"

    result = run_supported_normal_generation_v17(PROJECT_ROOT, artifact_root)

    assert result["status"] == "completed_unfit"
    assert result["confirmed_external_call_count"] == 0
    assert result["key_release_calibration"]["status"] == "nearest_unfit"
    assert result["key_release_calibration"]["selected_percent"] == 59
    assert result["generic_quality_passes"] == {
        "score": True,
        "pipeline": True,
        "rendered_texture_budget": True,
    }
    assert result["reference_fit"]["key_held_texture"] is False
    assert result["reference_fit"]["texture_fit"] is False
    assert result["promoted"] is False
    smf = artifact_root / "runs/whole-score/staged/final.mid"
    assert smf.is_file()
    assert result["smf"]["path"] == str(smf)
    run_dir = artifact_root / "runs/whole-score"
    state = json.loads((run_dir / "run-state.json").read_text(encoding="utf-8"))
    assert state["calls"]["confirmed_external_call_count"] == 0
    assert state["steps"]["rendered-output-validation"]["status"] == "completed"
    assert state["steps"]["publish-final"]["status"] == "completed"


def test_v17_rejects_a_changed_v16_score_before_creating_the_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original = v17_module.sha256_file

    def tampered(path: Path) -> str:
        if path.name == "score-spec.dsl":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(v17_module, "sha256_file", tampered)

    with pytest.raises(SupportedNormalGenerationV17Error, match="SHA-256"):
        run_supported_normal_generation_v17(PROJECT_ROOT, tmp_path / "v17")

    assert not (tmp_path / "v17").exists()
