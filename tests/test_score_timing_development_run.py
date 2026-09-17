from __future__ import annotations

import json
from pathlib import Path

import mido
import pytest

from llm_musical_composer.run_state import sha256_file
from llm_musical_composer.score_timing_development_run import (
    ScoreTimingDevelopmentRunError,
    pareto_candidate_ids,
    run_score_timing_development,
)


def _write_midi(path: Path, intervals: tuple[int, ...]) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    for index, interval in enumerate(intervals):
        track.append(
            mido.Message(
                "note_on",
                note=60 + index,
                velocity=64,
                channel=0,
                time=interval,
            )
        )
        track.append(
            mido.Message(
                "note_off",
                note=60 + index,
                velocity=0,
                channel=0,
                time=120,
            )
        )
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(path)


def _staging(tmp_path: Path) -> tuple[Path, Path]:
    staging = tmp_path / "development-smf"
    staging.mkdir()
    _write_midi(staging / "alpha.mid", (0, 120, 360, 120, 600))
    _write_midi(staging / "beta.mid", (0, 240, 120, 480, 240))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "development_smf": {
                    path.name: sha256_file(path) for path in sorted(staging.iterdir())
                },
            }
        ),
        encoding="utf-8",
    )
    return staging, manifest


def test_pareto_relation_keeps_tradeoffs_and_rejects_dominated_candidate() -> None:
    records = [
        {"candidate_id": "simple", "search_status": "complete", "metrics": {"a": 2, "b": 1}},
        {"candidate_id": "accurate", "search_status": "complete", "metrics": {"a": 1, "b": 2}},
        {"candidate_id": "dominated", "search_status": "complete", "metrics": {"a": 3, "b": 3}},
        {"candidate_id": "cycle", "search_status": "not_converged", "metrics": {"a": 0, "b": 0}},
    ]

    assert pareto_candidate_ids(records, metric_names=("a", "b")) == {
        "simple",
        "accurate",
    }


def test_development_runner_uses_only_staged_files_and_writes_all_candidates(
    tmp_path: Path,
) -> None:
    staging, manifest = _staging(tmp_path)
    output = tmp_path / "output"

    result = run_score_timing_development(
        staging_dir=staging,
        staging_manifest=manifest,
        output_dir=output,
    )

    assert result["status"] == "pass"
    assert result["source_count"] == 2
    assert result["candidate_count"] == 54
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["negative_control_count"] == 4
    assert 0 <= summary["source_with_viable_tradeoff_count"] <= 2
    run_spec = json.loads((output / "run-spec.json").read_text(encoding="utf-8"))
    assert set(run_spec["inputs"]) == {"staging_manifest", "development_smf"}
    assert "holdout" not in json.dumps(run_spec).lower()
    assert "withheld" not in json.dumps(run_spec).lower()
    records = [
        json.loads(line)
        for line in (output / "files.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(
        set(candidate["negative_control_comparison"])
        == {"observed-copy-negative-control", "uniform-grid-baseline"}
        for record in records
        for candidate in record["candidates"]
    )


def test_development_runner_rejects_unexpected_staged_file(tmp_path: Path) -> None:
    staging, manifest = _staging(tmp_path)
    _write_midi(staging / "unexpected.mid", (0, 120))

    with pytest.raises(ScoreTimingDevelopmentRunError, match="source set mismatch"):
        run_score_timing_development(
            staging_dir=staging,
            staging_manifest=manifest,
            output_dir=tmp_path / "output",
        )


def test_runner_preserves_generic_staging_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging, legacy_manifest = _staging(tmp_path)
    legacy = json.loads(legacy_manifest.read_text(encoding="utf-8"))
    manifest = tmp_path / "holdout-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "pass",
                "split_role": "holdout",
                "inputs": {"method_manifest": "f" * 64},
                "staged_smf": legacy["development_smf"],
            }
        ),
        encoding="utf-8",
    )
    method_manifest = tmp_path / "method-manifest.json"
    method_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "llm_musical_composer.score_timing_development_run.verify_method_manifest",
        lambda **_arguments: "f" * 64,
    )

    run_score_timing_development(
        staging_dir=staging,
        staging_manifest=manifest,
        output_dir=tmp_path / "output",
        repository_root=Path.cwd(),
        method_manifest=method_manifest,
    )

    run_spec = json.loads((tmp_path / "output" / "run-spec.json").read_text(encoding="utf-8"))
    assert run_spec["split_role"] == "holdout"
    assert "staged_smf" in run_spec["inputs"]
    assert "development_smf" not in run_spec["inputs"]
    assert run_spec["inputs"]["method_manifest"] == "f" * 64


def test_holdout_runner_requires_method_manifest_before_output(tmp_path: Path) -> None:
    staging, legacy_manifest = _staging(tmp_path)
    legacy = json.loads(legacy_manifest.read_text(encoding="utf-8"))
    manifest = tmp_path / "holdout-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "pass",
                "split_role": "holdout",
                "inputs": {"method_manifest": "f" * 64},
                "staged_smf": legacy["development_smf"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    with pytest.raises(ScoreTimingDevelopmentRunError, match="method manifest is required"):
        run_score_timing_development(
            staging_dir=staging,
            staging_manifest=manifest,
            output_dir=output,
        )

    assert not output.exists()
