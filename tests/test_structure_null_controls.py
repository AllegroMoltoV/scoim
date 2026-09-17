from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_musical_composer.structure_null_controls import (
    IDENTITY_PERMUTATION,
    block_permutations,
    boundary_statistics,
    build_null_control_summary,
    evaluate_record,
    main,
    permute_coordinates,
    write_structure_null_controls,
)

METRICS = (
    "onset_count",
    "duration_median",
    "velocity_median",
    "polyphony_mean",
    "polyphony_max",
    "pitch_median",
    "pitch_range",
    "ioi_median",
    "pitch_direction_change_rate",
)


def _windows(count: int, offset: float = 0.0) -> list[dict]:
    values = [1.0, 2.0, 7.0, 8.0] * (count // 4)
    return [
        {
            "index": index,
            "start": index / count,
            "end": (index + 1) / count,
            "center": (index + 0.5) / count,
            "features": {
                name: value + offset + metric_index / 10
                for metric_index, name in enumerate(METRICS)
            },
        }
        for index, value in enumerate(values)
    ]


def _report(count: int, offset: float = 0.0) -> dict:
    return {
        "status": "pass",
        "window_count": count,
        "coordinate": "elapsed_time",
        "phase_fraction": 0.0,
        "windows": _windows(count, offset),
        "novelty": {"consensus_candidates": []},
    }


def _record(name: str = "sample.mid") -> dict:
    record = {
        "name": name,
        "status": "pass",
        "coordinates": {
            "elapsed_time": {
                "resolutions": {"4": _report(4), "8": _report(8)},
            },
            "onset_order": {
                "resolutions": {"4": _report(4, 0.5), "8": _report(8, 0.5)},
            },
        },
        "persistent_boundaries": [],
    }
    record["persistent_boundaries"] = permute_coordinates(
        record["coordinates"], IDENTITY_PERMUTATION
    )["persistent_boundaries"]
    return record


def _feature_sequence(coordinates: dict, coordinate: str, resolution: str) -> list[dict]:
    return [
        window["features"]
        for window in coordinates[coordinate]["resolutions"][resolution]["windows"]
    ]


def test_all_four_block_permutations_are_unique_and_deterministic() -> None:
    first = block_permutations()
    second = block_permutations()

    assert first == second
    assert len(first) == 24
    assert len(set(first)) == 24
    assert first[0] == IDENTITY_PERMUTATION
    assert len([item for item in first if item != IDENTITY_PERMUTATION]) == 23


def test_permutation_preserves_feature_multiset_and_order_inside_each_block() -> None:
    original = _record()["coordinates"]
    changed = permute_coordinates(original, (1, 0, 3, 2))["coordinates"]

    for coordinate in original:
        for resolution in ("4", "8"):
            before = _feature_sequence(original, coordinate, resolution)
            after = _feature_sequence(changed, coordinate, resolution)
            assert sorted(json.dumps(item, sort_keys=True) for item in before) == sorted(
                json.dumps(item, sort_keys=True) for item in after
            )
            block_size = int(resolution) // 4
            assert after[:block_size] == before[block_size : block_size * 2]
            changed_starts = [
                window["start"]
                for window in changed[coordinate]["resolutions"][resolution]["windows"]
            ]
            original_starts = [
                window["start"]
                for window in original[coordinate]["resolutions"][resolution]["windows"]
            ]
            assert changed_starts == original_starts


def test_identity_recomputes_the_same_boundaries_as_the_saved_algorithm() -> None:
    record = _record()
    identity = evaluate_record(record)["permutations"][0]
    recomputed = permute_coordinates(record["coordinates"], IDENTITY_PERMUTATION)

    assert identity["permutation"] == list(IDENTITY_PERMUTATION)
    assert identity["persistent_boundaries"]
    assert identity["persistent_boundaries"] == recomputed["persistent_boundaries"]
    assert evaluate_record(record)["identity_matches_saved"] is True


def test_saved_supporting_metric_order_is_not_treated_as_a_musical_difference() -> None:
    record = _record()
    for boundary in record["persistent_boundaries"]:
        for evidence in boundary["evidence"]:
            evidence["supporting_metrics"].reverse()

    result = evaluate_record(record)

    assert result["status"] == "pass"
    assert result["identity_matches_saved"] is True


def test_changed_implementation_or_input_does_not_silently_compare_different_algorithms() -> None:
    record = _record()
    record["persistent_boundaries"] = []

    result = evaluate_record(record)

    assert result["status"] == "unable_to_investigate"
    assert result["error"]["type"] == "IdentityMismatch"


def test_unavailable_resolution_is_not_converted_to_zero_evidence() -> None:
    record = _record()
    unavailable = {
        "status": "unable_to_investigate",
        "detail": "not enough onsets",
        "windows": [],
    }
    record["coordinates"]["elapsed_time"]["resolutions"]["8"] = unavailable
    record["persistent_boundaries"] = permute_coordinates(
        record["coordinates"], IDENTITY_PERMUTATION
    )["persistent_boundaries"]

    result = evaluate_record(record)

    assert result["status"] == "pass"
    assert result["unavailable_resolution_count"] == 1
    for permutation in result["permutations"]:
        assert permutation["available_resolution_count"] == 3


def test_equal_permutation_statistics_fail_with_exact_p_one() -> None:
    summary = build_null_control_summary(
        [
            {
                "name": "equal.mid",
                "status": "pass",
                "permutations": [
                    {"statistics": {"extra_support": 5}} for _ in block_permutations()
                ],
            }
        ]
    )

    assert summary["exact_permutation_p"] == 1.0
    assert summary["decision"] == "fail"
    assert summary["boundary_use"] == "diagnostic_only"
    assert summary["generation_boundary_target"] is None


def test_identity_only_maximum_has_exact_p_one_over_twenty_four() -> None:
    permutations = [
        {"statistics": {"extra_support": 3}},
        *[{"statistics": {"extra_support": 1}} for _ in range(23)],
    ]
    summary = build_null_control_summary(
        [
            {"name": f"song-{index}.mid", "status": "pass", "permutations": permutations}
            for index in range(3)
        ]
    )

    assert summary["exact_permutation_p"] == pytest.approx(1 / 24)
    assert summary["paired_extra_support_difference"]["median"] == 2
    assert summary["decision"] == "pass_stage_one"
    assert summary["boundary_use"] == "diagnostic_only"
    assert summary["next_step"] == "phase_control"


def test_unavailable_record_remains_separate_from_zero_evidence() -> None:
    record = _record("broken.mid")
    record["status"] = "unable_to_investigate"
    record["error"] = {"type": "ValueError", "message": "broken input"}

    result = evaluate_record(record)

    assert result["status"] == "unable_to_investigate"
    assert result["error"] == record["error"]
    assert "permutations" not in result


def test_empty_boundary_statistics_and_unavailable_corpus_remain_explicit() -> None:
    statistics = boundary_statistics([])
    summary = build_null_control_summary(
        [{"name": "broken.mid", "status": "unable_to_investigate"}]
    )

    assert statistics["boundary_count"] == 0
    assert statistics["quartile_concentration_rate"] is None
    assert summary["status"] == "unable_to_investigate"
    assert summary["unable_to_investigate_count"] == 1


def test_invalid_coordinates_and_window_shapes_become_investigation_errors() -> None:
    missing = _record()
    missing["coordinates"] = {}
    assert evaluate_record(missing)["status"] == "unable_to_investigate"

    malformed = _record()
    malformed["coordinates"]["elapsed_time"]["resolutions"]["4"]["windows"] = []
    result = evaluate_record(malformed)
    assert result["status"] == "unable_to_investigate"
    assert result["error"]["type"] == "ValueError"


def test_incomplete_permutation_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="all 24"):
        build_null_control_summary([{"name": "short.mid", "status": "pass", "permutations": []}])


def test_writer_is_deterministic_and_never_writes_a_generation_target(tmp_path: Path) -> None:
    source = tmp_path / "files.jsonl"
    source.write_text(json.dumps(_record(), sort_keys=True) + "\n", encoding="utf-8")

    first = write_structure_null_controls(source, tmp_path / "first")
    second = write_structure_null_controls(source, tmp_path / "second")

    assert first == second
    first_manifest = json.loads((tmp_path / "first" / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads(
        (tmp_path / "second" / "manifest.json").read_text(encoding="utf-8")
    )
    assert first_manifest == second_manifest
    assert first["generation_boundary_target"] is None
    assert not (tmp_path / "first" / "generation-boundary-target.json").exists()
    assert set(first_manifest["sha256"]) == {
        "input_jsonl",
        "implementation",
        "files_jsonl",
        "summary_json",
    }


@pytest.mark.parametrize("content, message", [("{\n", "invalid JSONL"), ("[]\n", "not an object")])
def test_invalid_jsonl_is_not_treated_as_an_empty_corpus(
    tmp_path: Path, content: str, message: str
) -> None:
    source = tmp_path / "files.jsonl"
    source.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        write_structure_null_controls(source, tmp_path / "output")


def test_main_writes_results(tmp_path: Path) -> None:
    source = tmp_path / "files.jsonl"
    source.write_text(json.dumps(_record(), sort_keys=True) + "\n", encoding="utf-8")
    output = tmp_path / "output"

    status = main(["--records", str(source), "--output-dir", str(output)])

    assert status == 0
    assert (output / "summary.json").exists()
