from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mido
import pytest

from llm_musical_composer.reference_decomposition_run import (
    ReferenceDecompositionRunError,
    run_reference_decomposition,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_smf(path: Path, *, pitch: int, channel: int = 0) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.Message("control_change", channel=channel, control=1, value=3, time=0))
    track.append(mido.Message("control_change", channel=channel, control=64, value=127, time=0))
    track.append(mido.Message("note_on", channel=channel, note=pitch, velocity=80, time=0))
    track.append(mido.Message("note_off", channel=channel, note=pitch, velocity=7, time=480))
    track.append(mido.Message("control_change", channel=channel, control=64, value=0, time=0))
    midi.tracks.append(track)
    midi.save(path)


def _write_inputs(root: Path) -> dict[str, Path]:
    source = root / "source"
    source.mkdir()
    _write_smf(source / "first.mid", pitch=60)
    _write_smf(source / "second.mid", pitch=67, channel=3)
    records = [
        {"name": path.name, "sha256": _sha256(path)} for path in sorted(source.glob("*.mid"))
    ]
    reference_files = root / "reference-files.jsonl"
    reference_summary = root / "reference-summary.json"
    reference_files.write_text("\n".join(json.dumps(item) for item in records) + "\n")
    reference_summary.write_text(json.dumps({"status": "pass", "source_count": 2}))
    reference_manifest = root / "reference-manifest.json"
    reference_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "pass",
                "inputs": {item["name"]: item["sha256"] for item in records},
                "outputs": {
                    "files.jsonl": _sha256(reference_files),
                    "summary.json": _sha256(reference_summary),
                },
            }
        )
    )
    structure_files = root / "structure-files.jsonl"
    structure_summary = root / "structure-summary.json"
    structure_controls = root / "structure-controls.json"
    structure_files.write_text("{}\n")
    structure_summary.write_text(json.dumps({"source_count": 2}))
    structure_controls.write_text("{}")
    structure_manifest = root / "structure-manifest.json"
    structure_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_files": records,
                "artifact_sha256": {
                    "files.jsonl": _sha256(structure_files),
                    "summary.json": _sha256(structure_summary),
                    "controls.json": _sha256(structure_controls),
                },
            }
        )
    )
    audit_paths = []
    for name in (
        "audit-files.jsonl",
        "audit-summary.json",
        "audit-duplicates.json",
        "audit-groups.json",
    ):
        path = root / name
        path.write_text("{}\n")
        audit_paths.append(path)
    return {
        "source_dir": source,
        "reference_manifest": reference_manifest,
        "reference_files": reference_files,
        "reference_summary": reference_summary,
        "structure_manifest": structure_manifest,
        "structure_files": structure_files,
        "structure_summary": structure_summary,
        "structure_controls": structure_controls,
        "audit_paths": audit_paths,
        "known_source_run": _write_known_source_run(root),
    }


def _write_known_source_run(root: Path) -> Path:
    run = root / "known-source-run"
    outputs = run / "outputs"
    outputs.mkdir(parents=True)
    for name, content in (
        ("piece-plan.dsl", "piece"),
        ("score-spec.dsl", "score"),
        ("performance-spec.dsl", "performance"),
    ):
        (outputs / name).write_text(content)
    _write_smf(outputs / "final.mid", pitch=65)
    state = {
        "status": "completed",
        "steps": {
            "piece-plan": {
                "status": "completed",
                "outputs": {
                    "path": "outputs\\piece-plan.dsl",
                    "sha256": _sha256(outputs / "piece-plan.dsl"),
                },
            },
            "score-spec-aggregate": {
                "status": "completed",
                "outputs": {
                    "path": "outputs\\score-spec.dsl",
                    "sha256": _sha256(outputs / "score-spec.dsl"),
                },
            },
            "performance-spec": {
                "status": "completed",
                "outputs": {
                    "path": "outputs\\performance-spec.dsl",
                    "sha256": _sha256(outputs / "performance-spec.dsl"),
                },
            },
            "publish-final": {
                "status": "completed",
                "outputs": {
                    "smf_path": "outputs\\final.mid",
                    "smf_sha256": _sha256(outputs / "final.mid"),
                },
            },
        },
    }
    (run / "run-state.json").write_text(json.dumps(state))
    return run


def test_run_processes_every_manifest_source_and_records_event_support(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    output = tmp_path / "output"

    result = run_reference_decomposition(output_dir=output, **inputs)

    assert result["status"] == "pass"
    assert result["source_count"] == 2
    records = [
        json.loads(line) for line in (output / "corpus-observations.jsonl").read_text().splitlines()
    ]
    assert [record["name"] for record in records] == ["first.mid", "second.mid"]
    assert all(record["event_roundtrip_status"] == "pass" for record in records)
    assert all(record["performance_status"] == "assessed" for record in records)
    support = json.loads((output / "event-support.json").read_text())
    assert support["preserved"]["control_change"] == 6
    assert support["preserved"]["note_off"] == 2
    assert (
        json.loads((output / "loss-ledger.jsonl").read_text().splitlines()[0])[
            "unsupported_event_count"
        ]
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "pass"
    assert set(manifest["outputs"]) >= {
        "corpus-observations.jsonl",
        "destructive-controls.json",
        "event-support.json",
        "known-source-controls.json",
        "loss-ledger.jsonl",
        "roundtrip.jsonl",
        "result.json",
        "run-spec.json",
    }
    known_source = json.loads((output / "known-source-controls.json").read_text())
    assert known_source["status"] == "affirmative_evidence"
    assert known_source["observed"]["event_roundtrip_status"] == "pass"
    assert set(known_source["verified_outputs"]) == {
        "performance_spec",
        "piece_plan",
        "score_spec",
        "smf",
    }
    destructive = json.loads((output / "destructive-controls.json").read_text())
    assert destructive["phase_1_status"] == "fixture_defined"
    assert {control["name"] for control in destructive["controls"]} >= {
        "pitch_change",
        "timing_change",
        "pedal_removal",
    }


def test_run_rejects_source_hash_mismatch_before_creating_results(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    (inputs["source_dir"] / "first.mid").write_bytes(b"not midi")
    output = tmp_path / "output"

    with pytest.raises(ReferenceDecompositionRunError, match="SHA-256 mismatch"):
        run_reference_decomposition(output_dir=output, **inputs)

    assert not (output / "result.json").exists()


def test_run_rejects_manifest_artifact_hash_mismatch(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    inputs["structure_summary"].write_text("changed")

    with pytest.raises(ReferenceDecompositionRunError, match="artifact SHA-256 mismatch"):
        run_reference_decomposition(output_dir=tmp_path / "output", **inputs)


def test_run_rejects_known_source_hash_mismatch(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    known_source_run = inputs["known_source_run"]
    (known_source_run / "outputs" / "score-spec.dsl").write_text("changed")

    with pytest.raises(
        ReferenceDecompositionRunError, match="known source score_spec SHA-256 mismatch"
    ):
        run_reference_decomposition(output_dir=tmp_path / "output", **inputs)
