from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_musical_composer.style_target import (
    StyleTargetRequest,
    build_style_target,
    evaluate_style_features,
    evaluate_style_target_feasibility,
    extract_style_features,
    main,
    write_style_target,
)


def record(
    name: str,
    *,
    density: float,
    polyphony: float,
    velocity: float,
    register: float,
) -> dict:
    onset_counts = [density * factor for factor in (0.7, 1.0, 1.4, 0.9)]
    windows = []
    for index in range(4):
        windows.append(
            {
                "index": index,
                "start": index / 4,
                "end": (index + 1) / 4,
                "center": (index + 0.5) / 4,
                "features": {
                    "onset_count": onset_counts[index],
                    "polyphony_mean": polyphony + (-0.2, 0.0, 0.4, -0.1)[index],
                    "velocity_median": velocity + (-4.0, 0.0, 6.0, -2.0)[index],
                    "pitch_median": register + (-3.0, 0.0, 5.0, -1.0)[index],
                },
            }
        )
    return {
        "name": name,
        "status": "pass",
        "note_count": round(sum(onset_counts)),
        "duration_ms": 40_000,
        "coordinates": {
            "elapsed_time": {
                "resolutions": {
                    "4": {
                        "status": "pass",
                        "windows": windows,
                    }
                }
            }
        },
    }


def corpus() -> list[dict]:
    return [
        record("calm.mid", density=20, polyphony=1.2, velocity=55, register=55),
        record("middle.mid", density=40, polyphony=2.0, velocity=75, register=64),
        record("bright.mid", density=70, polyphony=3.1, velocity=100, register=76),
        record("dense.mid", density=100, polyphony=4.0, velocity=110, register=68),
    ]


def test_explicit_anchor_is_case_insensitive_and_has_zero_axis_distances() -> None:
    target = build_style_target(
        corpus(),
        StyleTargetRequest(anchor_name="MIDDLE.MID", neighbor_count=3),
    )

    assert target["selection"]["mode"] == "explicit_anchor"
    assert target["selection"]["anchor_name"] == "middle.mid"
    anchor = next(item for item in target["neighbors"] if item["name"] == "middle.mid")
    assert set(anchor["axis_distances"].values()) == {0.0}
    assert set(target["axes"]) == {"density", "polyphony", "velocity", "register"}
    assert target["neighbor_count"] == 3


def test_missing_excluded_duplicate_and_invalid_requests_are_rejected() -> None:
    with pytest.raises(ValueError, match="not found"):
        build_style_target(corpus(), StyleTargetRequest(anchor_name="missing.mid"))
    with pytest.raises(ValueError, match="excluded"):
        build_style_target(corpus(), StyleTargetRequest(anchor_name="rut.mid"))
    with pytest.raises(ValueError, match="duplicate"):
        build_style_target(
            [
                *corpus(),
                record("CALM.MID", density=10, polyphony=1, velocity=40, register=50),
            ],
            StyleTargetRequest(anchor_name="calm.mid"),
        )
    with pytest.raises(ValueError, match="0 and 1"):
        StyleTargetRequest(density=1.1)
    with pytest.raises(ValueError, match="finite"):
        StyleTargetRequest(density=float("nan"))
    with pytest.raises(ValueError, match="neighbor_count"):
        StyleTargetRequest(neighbor_count=2)
    with pytest.raises(ValueError, match="cannot be combined"):
        StyleTargetRequest(anchor_name="calm.mid", density=0.5)


def test_auto_controls_choose_deterministically_without_using_shapes_as_input() -> None:
    request = StyleTargetRequest(
        density=1.0,
        polyphony=1.0,
        velocity=1.0,
        register=0.5,
        neighbor_count=3,
    )

    first = build_style_target(corpus(), request)
    second = build_style_target(reversed(corpus()), request)

    assert first == second
    assert first["selection"]["mode"] == "automatic_controls"
    assert first["selection"]["anchor_name"] == "dense.mid"
    assert set(first["selection"]["control_axis_distances"]) == {
        "density",
        "polyphony",
        "velocity",
        "register",
    }


def test_neighbor_shortage_is_not_silently_reduced() -> None:
    with pytest.raises(ValueError, match="usable reference count"):
        build_style_target(corpus(), StyleTargetRequest(anchor_name="middle.mid", neighbor_count=7))


def test_unavailable_anchor_reports_its_state_instead_of_not_found() -> None:
    invalid = record("invalid.mid", density=40, polyphony=2, velocity=70, register=60)
    invalid["coordinates"]["elapsed_time"]["resolutions"]["4"]["status"] = "unable_to_investigate"

    with pytest.raises(ValueError, match="unable_to_investigate"):
        build_style_target(
            [*corpus(), invalid],
            StyleTargetRequest(anchor_name="invalid.mid", neighbor_count=3),
        )


def test_each_level_change_affects_only_its_matching_raw_axis() -> None:
    base = extract_style_features(
        record("base.mid", density=40, polyphony=2, velocity=70, register=60)
    )
    changed = extract_style_features(
        record("changed.mid", density=80, polyphony=2, velocity=70, register=60)
    )

    assert base["density"] != changed["density"]
    for axis in ("polyphony", "velocity", "register"):
        assert base[axis] == changed[axis]


def test_transposition_changes_only_register_axis() -> None:
    original_record = record("original.mid", density=40, polyphony=2, velocity=70, register=60)
    transposed_record = json.loads(json.dumps(original_record))
    transposed_record["name"] = "transposed.mid"
    for window in transposed_record["coordinates"]["elapsed_time"]["resolutions"]["4"]["windows"]:
        window["features"]["pitch_median"] += 12

    original = extract_style_features(original_record)
    transposed = extract_style_features(transposed_record)

    assert original["register"] != transposed["register"]
    for axis in ("density", "polyphony", "velocity"):
        assert original[axis] == transposed[axis]


def test_time_stretch_changes_density_level_but_preserves_four_part_shapes() -> None:
    original_record = record("original.mid", density=40, polyphony=2, velocity=70, register=60)
    stretched_record = json.loads(json.dumps(original_record))
    stretched_record["name"] = "stretched.mid"
    stretched_record["duration_ms"] *= 2

    original = extract_style_features(original_record)
    stretched = extract_style_features(stretched_record)

    assert original["density"]["level"] == stretched["density"]["level"] * 2
    for axis in ("density", "polyphony", "velocity", "register"):
        assert original[axis]["shape"] == stretched[axis]["shape"]


def test_equal_auto_distances_use_casefolded_name_as_tie_break() -> None:
    tied = [
        record("z.mid", density=40, polyphony=2, velocity=70, register=60),
        record("a.mid", density=40, polyphony=2, velocity=70, register=60),
        record("m.mid", density=40, polyphony=2, velocity=70, register=60),
    ]

    target = build_style_target(tied, StyleTargetRequest(neighbor_count=3))

    assert target["selection"]["anchor_name"] == "a.mid"
    assert [item["name"] for item in target["neighbors"]] == ["a.mid", "m.mid", "z.mid"]


def test_missing_four_window_report_is_unable_instead_of_normal_values() -> None:
    invalid = record("invalid.mid", density=40, polyphony=2, velocity=70, register=60)
    invalid["coordinates"]["elapsed_time"]["resolutions"]["4"]["status"] = "unable_to_investigate"

    with pytest.raises(ValueError, match="elapsed-time four-window"):
        extract_style_features(invalid)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda item: item.update(status="unable_to_investigate"), "record is unavailable"),
        (
            lambda item: item["coordinates"]["elapsed_time"]["resolutions"]["4"][
                "windows"
            ].reverse(),
            "window indices",
        ),
        (
            lambda item: item["coordinates"]["elapsed_time"]["resolutions"]["4"]["windows"][0][
                "features"
            ].update(onset_count=float("nan")),
            "finite",
        ),
        (
            lambda item: item["coordinates"]["elapsed_time"]["resolutions"]["4"]["windows"][0][
                "features"
            ].update(onset_count=-1),
            "non-negative",
        ),
    ],
)
def test_corrupt_records_do_not_become_normal_features(mutate, message) -> None:
    invalid = record("invalid.mid", density=40, polyphony=2, velocity=70, register=60)
    mutate(invalid)

    with pytest.raises(ValueError, match=message):
        extract_style_features(invalid)


def test_incomplete_zero_and_inconsistent_counts_are_rejected() -> None:
    incomplete = record("incomplete.mid", density=40, polyphony=2, velocity=70, register=60)
    del incomplete["coordinates"]["elapsed_time"]["resolutions"]["4"]["windows"][0]["features"][
        "polyphony_mean"
    ]
    with pytest.raises(ValueError, match="incomplete"):
        extract_style_features(incomplete)

    zero_duration = record("zero.mid", density=40, polyphony=2, velocity=70, register=60)
    zero_duration["duration_ms"] = 0
    with pytest.raises(ValueError, match="positive duration"):
        extract_style_features(zero_duration)

    inconsistent = record("count.mid", density=40, polyphony=2, velocity=70, register=60)
    inconsistent["note_count"] += 10
    with pytest.raises(ValueError, match="differs"):
        extract_style_features(inconsistent)


def test_prompt_target_contains_ranges_but_no_names_notes_boundaries_or_repetition() -> None:
    target = build_style_target(
        corpus(),
        StyleTargetRequest(anchor_name="middle.mid", neighbor_count=3),
    )
    serialized = json.dumps(target["prompt_target"], ensure_ascii=False).casefold()

    assert "middle.mid" not in serialized
    assert "note" not in serialized
    assert "boundar" not in serialized
    assert "repetition" not in serialized
    assert target["axes"]["density"]["level"]["available_count"] == 3
    assert set(target["axes"]["density"]["level"]) >= {
        "minimum",
        "p25",
        "median",
        "p75",
        "maximum",
    }
    assert set(target["prompt_target"]["axis_targets"]["density"]) == {"level"}
    assert "curve" not in serialized
    assert "four_part_shape" not in serialized


def test_candidate_evaluation_keeps_axis_results_separate() -> None:
    target = build_style_target(
        corpus(),
        StyleTargetRequest(anchor_name="middle.mid", neighbor_count=3),
    )
    features = extract_style_features(
        record("candidate.mid", density=40, polyphony=2, velocity=75, register=64)
    )

    evaluation = evaluate_style_features(features, target)

    assert set(evaluation["prompted_level"]["axes"]) == {
        "density",
        "polyphony",
        "velocity",
        "register",
    }
    assert set(evaluation["diagnostic_shape"]["axes"]) == {
        "density",
        "polyphony",
        "velocity",
        "register",
    }
    assert "total_loss" not in evaluation
    assert evaluation["prompted_level"]["worst_axis_distance"] >= 0
    assert evaluation["diagnostic_shape"]["semantics"] == (
        "four equal elapsed-time observations not shown to the generator"
    )

    extreme = json.loads(json.dumps(features))
    extreme["density"]["level"] = -100
    extreme["velocity"]["level"] = 1_000
    outside = evaluate_style_features(extreme, target)
    assert outside["prompted_level"]["axes"]["density"]["normalized_distance"] > 0
    assert outside["prompted_level"]["axes"]["velocity"]["normalized_distance"] > 0


def test_selected_anchor_is_always_inside_its_own_target_ranges() -> None:
    target = build_style_target(
        corpus(),
        StyleTargetRequest(
            density=0.0,
            polyphony=0.0,
            velocity=0.0,
            register=0.0,
            neighbor_count=3,
        ),
    )
    anchor_name = target["selection"]["anchor_name"]
    anchor_record = next(item for item in corpus() if item["name"] == anchor_name)

    evaluation = evaluate_style_features(extract_style_features(anchor_record), target)

    assert evaluation["prompted_level"]["inside_axis_count"] == 4
    assert evaluation["prompted_level"]["worst_axis_distance"] == 0.0


def test_shape_outlier_does_not_turn_a_prompted_level_pass_into_failure() -> None:
    target = build_style_target(
        corpus(),
        StyleTargetRequest(anchor_name="middle.mid", neighbor_count=3),
    )
    features = extract_style_features(
        record("candidate.mid", density=40, polyphony=2, velocity=75, register=64)
    )
    for axis in features.values():
        axis["shape"] = [-10_000, 10_000, 20_000, -20_000]

    evaluation = evaluate_style_features(features, target)

    assert evaluation["prompted_level"]["inside_axis_count"] == 4
    assert evaluation["diagnostic_shape"]["inside_axis_count"] == 0


def test_density_feasibility_reports_required_notes_and_conflict() -> None:
    prompt_target = {
        "axis_targets": {"density": {"level": {"range_low": 6.12745098, "range_high": 12.54863372}}}
    }

    report = evaluate_style_target_feasibility(
        prompt_target,
        duration_ms=180_000,
        max_note_count=950,
    )

    assert report["status"] == "conflict"
    assert report["checks"]["density_level"] == {
        "status": "unreachable",
        "target_range_low": 6.12745098,
        "target_range_high": 12.54863372,
        "maximum_reachable_level": 5.27777778,
        "minimum_required_note_count": 1103,
        "shortfall_note_count": 153,
    }


def test_density_feasibility_passes_when_the_range_is_reachable() -> None:
    prompt_target = {"axis_targets": {"density": {"level": {"range_low": 4.0, "range_high": 6.0}}}}

    report = evaluate_style_target_feasibility(
        prompt_target,
        duration_ms=180_000,
        max_note_count=950,
    )

    assert report["status"] == "pass"
    assert report["checks"]["density_level"]["status"] == "reachable"


@pytest.mark.parametrize(
    "prompt_target, duration_ms, max_note_count, message",
    [
        (
            {"axis_targets": {"density": {"level": {"range_low": 1, "range_high": 2}}}},
            0,
            10,
            "duration_ms",
        ),
        (
            {"axis_targets": {"density": {"level": {"range_low": 1, "range_high": 2}}}},
            1_000,
            -1,
            "max_note_count",
        ),
        ({}, 1_000, 10, "no density level range"),
        (
            {"axis_targets": {"density": {"level": {"range_low": float("nan"), "range_high": 2}}}},
            1_000,
            10,
            "must be finite",
        ),
        (
            {"axis_targets": {"density": {"level": {"range_low": 3, "range_high": 2}}}},
            1_000,
            10,
            "range is invalid",
        ),
    ],
)
def test_density_feasibility_rejects_invalid_inputs(
    prompt_target, duration_ms, max_note_count, message
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_style_target_feasibility(
            prompt_target,
            duration_ms=duration_ms,
            max_note_count=max_note_count,
        )


def test_excluded_and_unusable_records_are_listed_without_becoming_neighbors() -> None:
    excluded = record("rut.mid", density=40, polyphony=2, velocity=70, register=60)
    invalid = record("invalid.mid", density=40, polyphony=2, velocity=70, register=60)
    invalid["status"] = "unable_to_investigate"

    target = build_style_target(
        [*corpus(), excluded, invalid],
        StyleTargetRequest(anchor_name="middle.mid", neighbor_count=3),
    )

    statuses = {item["name"]: item["status"] for item in target["unable_to_investigate"]}
    assert statuses == {"invalid.mid": "unable_to_investigate", "rut.mid": "excluded"}
    assert {item["name"] for item in target["neighbors"]}.isdisjoint(statuses)


def test_fewer_than_two_usable_records_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least two usable"):
        build_style_target(
            [record("only.mid", density=40, polyphony=2, velocity=70, register=60)],
            StyleTargetRequest(anchor_name="only.mid", neighbor_count=3),
        )


def test_write_style_target_is_deterministic_and_records_hashes(tmp_path: Path) -> None:
    records_path = tmp_path / "files.jsonl"
    records_path.write_text(
        "\n" + "".join(json.dumps(item, sort_keys=True) + "\n" for item in corpus()),
        encoding="utf-8",
    )
    request = StyleTargetRequest(anchor_name="middle.mid", neighbor_count=3)

    first = write_style_target(records_path, tmp_path / "first", request)
    second = write_style_target(records_path, tmp_path / "second", request)

    assert first == second
    first_manifest = json.loads((tmp_path / "first" / "manifest.json").read_text())
    second_manifest = json.loads((tmp_path / "second" / "manifest.json").read_text())
    assert first_manifest == second_manifest
    assert set(first_manifest["sha256"]) == {
        "input_jsonl",
        "implementation",
        "request",
        "style_target",
    }


@pytest.mark.parametrize("content, message", [("{\n", "invalid JSONL"), ("[]\n", "not an object")])
def test_invalid_jsonl_is_not_treated_as_an_empty_corpus(
    tmp_path: Path, content: str, message: str
) -> None:
    records_path = tmp_path / "files.jsonl"
    records_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        write_style_target(records_path, tmp_path / "output", StyleTargetRequest())


def test_main_writes_an_explicit_anchor_target(tmp_path: Path) -> None:
    records_path = tmp_path / "files.jsonl"
    records_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in corpus()),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    status = main(
        [
            "--records",
            str(records_path),
            "--output-dir",
            str(output_dir),
            "--anchor",
            "middle.mid",
            "--neighbor-count",
            "3",
        ]
    )

    assert status == 0
    saved = json.loads((output_dir / "style-target.json").read_text(encoding="utf-8"))
    assert saved["selection"]["anchor_name"] == "middle.mid"
