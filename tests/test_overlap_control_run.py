from __future__ import annotations

import json
import shutil
from pathlib import Path

import mido
import pytest

from llm_musical_composer.overlap_control_run import (
    EXPECTED_BASELINE_HASHES,
    run_overlap_control,
)

ROOT = Path(__file__).parents[1]
INPUT_DIR = ROOT / ".appendix" / "multiscale-calibration-run-v8" / "inputs"
BASELINE_DIR = ROOT / ".appendix" / "control-reference-baseline-v3"


def test_run_writes_three_measured_overlap_outputs(tmp_path: Path) -> None:
    result = run_overlap_control(output_root=tmp_path / "runs")
    run_dir = Path(result["run_dir"])

    assert result["status"] == "machine_passed_with_unreachable"
    assert result["candidate_count"] == 18
    assert [item["level"] for item in result["outputs"]] == [
        "low",
        "middle",
        "high",
    ]
    assert [item["resolution"]["status"] for item in result["outputs"]] == [
        "unreachable",
        "quantized",
        "unreachable",
    ]
    assert all(item["safe"] for item in result["outputs"])
    overlaps = [item["controls"]["raw"]["重なり"] for item in result["outputs"]]
    assert overlaps == sorted(overlaps)
    assert len({item["controls"]["raw"]["発音頻度"] for item in result["outputs"]}) == 1
    for item in result["outputs"]:
        midi_path = run_dir / item["smf"]
        assert midi_path.is_file()
        assert mido.MidiFile(midi_path).length == pytest.approx(180.0, abs=0.001)
    assert (run_dir / "result.json").is_file()
    assert (run_dir / "manifest.json").is_file()


def test_run_is_deterministic_and_records_all_candidates(tmp_path: Path) -> None:
    first = run_overlap_control(output_root=tmp_path / "runs")
    second = run_overlap_control(output_root=tmp_path / "runs")

    assert second == first
    assert len(first["candidates"]) == 18
    assert all(candidate["safe"] for candidate in first["candidates"])
    run_dir = Path(first["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == run_dir.name
    assert manifest["baseline_hashes"] == EXPECTED_BASELINE_HASHES
    assert set(manifest["outputs"]) == {
        "outputs/high.mid",
        "outputs/low.mid",
        "outputs/middle.mid",
        "result.json",
    }


def test_changed_input_or_baseline_stops_before_new_run_directory(tmp_path: Path) -> None:
    copied_inputs = tmp_path / "inputs"
    shutil.copytree(INPUT_DIR, copied_inputs)
    with (copied_inputs / "score.music.py").open("a", encoding="utf-8") as output:
        output.write("\n")
    output_root = tmp_path / "input-runs"
    with pytest.raises(ValueError, match="input hash mismatch"):
        run_overlap_control(output_root=output_root, input_dir=copied_inputs)
    assert not output_root.exists()

    copied_baseline = tmp_path / "baseline"
    shutil.copytree(BASELINE_DIR, copied_baseline)
    with (copied_baseline / "summary.json").open("a", encoding="utf-8") as output:
        output.write("\n")
    output_root = tmp_path / "baseline-runs"
    with pytest.raises(ValueError, match="baseline hash mismatch"):
        run_overlap_control(
            output_root=output_root,
            summary_path=copied_baseline / "summary.json",
            baseline_manifest_path=copied_baseline / "manifest.json",
        )
    assert not output_root.exists()
