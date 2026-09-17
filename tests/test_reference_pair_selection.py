from pathlib import Path

from llm_musical_composer.reference_pair_selection import (
    descriptor_separation,
    select_reference_pair,
    write_reference_pair_selection,
)
from llm_musical_composer.run_state import sha256_json

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / ".appendix" / "reference-profile-v1"
CONTROL_DIR = ROOT / ".appendix" / "control-reference-baseline-v3"


def test_verified_artifacts_select_pair_with_separated_prompt_targets() -> None:
    result = select_reference_pair(REFERENCE_DIR, CONTROL_DIR)

    assert result.artifact["selected"]["names"] == [
        "TasteOfFall.mid",
        "UnderTheColdSkyIlluminated.mid",
    ]
    assert result.artifact["eligible_pair_count"] == 17
    assert result.artifact["fully_separated_pair_count"] == 13
    assert result.artifact["selected"]["separated_descriptor_count"] == 10
    assert result.artifact["selected"]["separated_groups"] == [
        "performance_texture",
        "pitch_harmony",
        "rhythm_time",
    ]
    assert result.artifact["target_errors"] == {
        "2f6-1.mid": "reference distribution is invalid: "
        "pitch_harmony.vertical_interval_class"
    }
    assert set(result.artifact["input_sha256"]) == {
        "reference_manifest",
        "reference_records",
        "reference_summary",
        "control_manifest",
        "control_records",
        "control_summary",
    }
    assert result.sha256 == sha256_json(result.artifact)


def test_pair_selection_is_deterministic() -> None:
    first = select_reference_pair(REFERENCE_DIR, CONTROL_DIR)
    second = select_reference_pair(REFERENCE_DIR, CONTROL_DIR)

    assert second == first


def test_selection_artifact_is_written_with_its_hash(tmp_path: Path) -> None:
    output = tmp_path / "selection.json"

    result = write_reference_pair_selection(REFERENCE_DIR, CONTROL_DIR, output)

    assert output.is_file()
    assert (tmp_path / "selection.sha256").read_text(encoding="ascii") == result.sha256 + "\n"


def test_descriptor_separation_preserves_signed_direction_and_overlap() -> None:
    first = {
        "id": "rhythm_time.example",
        "kind": "distribution",
        "neighborhood_center": [0.8, 0.2],
        "neighborhood_radius": 0.2,
    }
    second = {
        "id": "rhythm_time.example",
        "kind": "distribution",
        "neighborhood_center": [0.2, 0.8],
        "neighborhood_radius": 0.2,
    }

    separated = descriptor_separation(first, second)

    assert separated == {
        "id": "rhythm_time.example",
        "group": "rhythm_time",
        "kind": "distribution",
        "center_delta": [-0.6, 0.6],
        "center_distance": 0.6,
        "radius_sum": 0.4,
        "separation_margin": 0.2,
        "separated": True,
    }

    overlapping = descriptor_separation(
        {**first, "neighborhood_radius": 0.4},
        {**second, "neighborhood_radius": 0.3},
    )
    assert overlapping["separated"] is False
    assert overlapping["separation_margin"] == -0.1
