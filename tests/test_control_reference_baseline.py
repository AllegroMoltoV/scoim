from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from llm_musical_composer.control_reference_baseline import (
    AXES,
    BaselineError,
    build_axis_summary,
    extract_control_observables,
    normalize_record,
    normalize_value,
    validate_reference_inputs,
    write_baseline_artifacts,
)
from llm_musical_composer.reference_profile import ReferencePiece


def _piece(*, transpose: int = 0, reverse: bool = False) -> ReferencePiece:
    notes = [
        {"pitch": 60 + transpose, "onset_ms": 0, "duration_ms": 1000, "velocity": 72},
        {"pitch": 64 + transpose, "onset_ms": 0, "duration_ms": 500, "velocity": 68},
        {"pitch": 67 + transpose, "onset_ms": 500, "duration_ms": 1000, "velocity": 76},
    ]
    if reverse:
        notes.reverse()
    return ReferencePiece.from_dicts(name="fixture.mid", notes=notes, pedals=[])


def _timed_piece(onsets: list[int]) -> ReferencePiece:
    return ReferencePiece.from_dicts(
        name="timed-fixture.mid",
        notes=(
            {
                "pitch": 60 + index * 4,
                "onset_ms": onset,
                "duration_ms": 1000,
                "velocity": 72,
            }
            for index, onset in enumerate(onsets)
        ),
        pedals=[],
    )


def _raw_record(name: str, value: float) -> dict[str, object]:
    return {"name": name, "raw": {axis: value for axis in AXES}}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_observables_are_order_invariant_and_measure_both_density_directions() -> None:
    first = extract_control_observables(_piece())
    second = extract_control_observables(_piece(reverse=True))

    assert first == second
    assert first["raw"]["高さ"] == pytest.approx(191 / 3)
    assert first["raw"]["重なり"] == pytest.approx(5 / 3)
    assert first["raw"]["発音頻度"] == pytest.approx(4 / 3)
    assert first["diagnostics"]["height"] == {
        "maximum_midi_pitch": 67.0,
        "median_midi_pitch": 64.0,
        "minimum_midi_pitch": 60.0,
        "p10_midi_pitch": 60.8,
        "p90_midi_pitch": 66.4,
    }
    assert first["diagnostics"]["overlap"]["attack_size_distribution"] == {
        "1": 0.5,
        "2": 0.5,
    }
    assert first["diagnostics"]["overlap"]["active_polyphony_duration_distribution"] == {
        "1": pytest.approx(1 / 3),
        "2": pytest.approx(2 / 3),
    }
    assert first["diagnostics"]["overlap"]["mean_active_polyphony"] == pytest.approx(5 / 3)
    assert first["diagnostics"]["overlap"]["notes_per_attack"] == pytest.approx(1.5)
    assert first["diagnostics"]["attack_frequency"]["median_ioi_ms"] == 500.0
    assert first["diagnostics"]["attack_frequency"]["notes_per_second"] == pytest.approx(2.0)
    assert first["diagnostics"]["attack_group_count"] == 2
    assert first["diagnostics"]["exact_onset_count"] == 2
    assert first["diagnostics"]["attack_frequency"]["attack_group_tolerance_ms"] == 30


def test_attack_groups_use_anchor_time_without_transitive_chaining() -> None:
    observables = extract_control_observables(_timed_piece([0, 20, 40]))

    assert observables["diagnostics"]["attack_group_count"] == 2
    assert observables["diagnostics"]["exact_onset_count"] == 3
    assert observables["raw"]["発音頻度"] == pytest.approx(2 / 1.04)
    assert observables["diagnostics"]["overlap"]["attack_size_distribution"] == {
        "1": 0.5,
        "2": 0.5,
    }
    assert observables["diagnostics"]["overlap"]["notes_per_attack"] == 1.5
    assert observables["diagnostics"]["attack_frequency"]["median_ioi_ms"] == 40.0


def test_attack_groups_include_30_ms_boundary_but_not_31_ms() -> None:
    observables = extract_control_observables(_timed_piece([0, 30, 31]))

    assert observables["diagnostics"]["attack_group_count"] == 2
    assert observables["diagnostics"]["exact_onset_count"] == 3
    assert observables["diagnostics"]["attack_frequency"]["median_ioi_ms"] == 31.0


def test_single_attack_group_has_no_inter_onset_interval() -> None:
    observables = extract_control_observables(_timed_piece([0, 20]))

    assert observables["diagnostics"]["attack_group_count"] == 1
    assert observables["diagnostics"]["exact_onset_count"] == 2
    assert observables["diagnostics"]["attack_frequency"]["median_ioi_ms"] is None


def test_brightness_observable_is_transposition_invariant() -> None:
    original = extract_control_observables(_piece())
    transposed = extract_control_observables(_piece(transpose=5))

    assert original["raw"]["あかるさ"] == transposed["raw"]["あかるさ"]
    assert (
        original["diagnostics"]["brightness"]["modal_degree_balance"]
        == transposed["diagnostics"]["brightness"]["modal_degree_balance"]
    )


def test_axis_summary_maps_endpoints_and_midpoint_exactly() -> None:
    records = [_raw_record("high.mid", 10.0), _raw_record("low.mid", 0.0)]
    summary = build_axis_summary(records)

    for axis in AXES:
        assert summary[axis]["minimum"] == 0.0
        assert summary[axis]["maximum"] == 10.0
        assert summary[axis]["endpoints"] == {
            "maximum": ["high.mid"],
            "minimum": ["low.mid"],
        }
        assert normalize_value(0.0, minimum=0.0, maximum=10.0) == -1.0
        assert normalize_value(5.0, minimum=0.0, maximum=10.0) == 0.0
        assert normalize_value(10.0, minimum=0.0, maximum=10.0) == 1.0
    assert summary["あかるさ"]["input_domain"] == [-1, 0, 1]
    assert summary["あかるさ"]["input_kind"] == "integer"
    assert summary["重なり"]["raw_metric"] == "mean_active_polyphony"
    assert summary["発音頻度"]["raw_metric"] == "attack_group_count_per_second_30ms"


def test_axis_summary_records_all_tied_endpoints_deterministically() -> None:
    records = [
        _raw_record("b.mid", 0.0),
        _raw_record("A.mid", 0.0),
        _raw_record("z.mid", 1.0),
        _raw_record("Z.mid", 1.0),
    ]
    summary = build_axis_summary(records)

    for axis in AXES:
        assert summary[axis]["endpoints"]["minimum"] == ["A.mid", "b.mid"]
        assert summary[axis]["endpoints"]["maximum"] == ["Z.mid", "z.mid"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_normalization_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(BaselineError, match="finite"):
        normalize_value(value, minimum=0.0, maximum=1.0)


def test_axis_summary_rejects_zero_range() -> None:
    with pytest.raises(BaselineError, match="zero range"):
        build_axis_summary([_raw_record("a.mid", 1.0), _raw_record("b.mid", 1.0)])


def test_observation_is_not_clipped_outside_reference_range() -> None:
    summary = build_axis_summary([_raw_record("low.mid", 0.0), _raw_record("high.mid", 10.0)])
    observation = normalize_record(_raw_record("candidate.mid", 15.0), summary, observation=True)

    assert observation["normalized"] == {axis: 2.0 for axis in AXES}
    assert observation["range_status"] == {axis: "above" for axis in AXES}


def test_manifest_validation_uses_only_manifest_and_allows_excluded_files_on_disk(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    inputs: dict[str, str] = {}
    for name, payload in (("one.mid", b"one"), ("two.mid", b"two")):
        path = source / name
        path.write_bytes(payload)
        inputs[name] = _sha256(path)
    (source / "rut.mid").write_bytes(b"excluded")
    (source / "aimusic01.mid").write_bytes(b"excluded")

    validated = validate_reference_inputs({"inputs": inputs}, source, expected_count=2)

    assert [item.name for item in validated] == ["one.mid", "two.mid"]


def test_manifest_validation_rejects_excluded_name_missing_file_and_hash_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    good = source / "good.mid"
    good.write_bytes(b"good")

    with pytest.raises(BaselineError, match="excluded"):
        validate_reference_inputs({"inputs": {"rut.mid": _sha256(good)}}, source, expected_count=1)
    with pytest.raises(BaselineError, match="missing"):
        validate_reference_inputs(
            {"inputs": {"missing.mid": _sha256(good)}}, source, expected_count=1
        )
    with pytest.raises(BaselineError, match="hash mismatch"):
        validate_reference_inputs({"inputs": {"good.mid": "0" * 64}}, source, expected_count=1)


def test_artifact_writes_are_byte_reproducible_and_do_not_invent_voice_metrics(
    tmp_path: Path,
) -> None:
    record = extract_control_observables(_piece())
    other = extract_control_observables(_piece(transpose=5))
    other["name"] = "other.mid"
    for axis in AXES:
        record["raw"][axis] = 0.0
        other["raw"][axis] = 1.0
    summary = build_axis_summary([record, other])
    records = [
        normalize_record(record, summary),
        normalize_record(other, summary),
    ]
    provenance = {
        "axis_version": "four-control-corpus-minmax-v3",
        "environment": {"mido": "test", "python": "test"},
        "implementations": {"module.py": "a" * 64},
        "inputs": {"fixture.mid": "b" * 64, "other.mid": "c" * 64},
        "reference_manifest_sha256": "d" * 64,
    }
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_baseline_artifacts(
        first,
        records=records,
        summary=summary,
        observations=[],
        provenance=provenance,
    )
    write_baseline_artifacts(
        second,
        records=records,
        summary=summary,
        observations=[],
        provenance=provenance,
    )

    for name in ("records.jsonl", "summary.json", "observations.jsonl", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    payload = json.loads((first / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "voice_specific_pitch_center" in {item["id"] for item in payload["unavailable_features"]}
    assert not any("voice" in key for key in payload["diagnostics"]["height"])
