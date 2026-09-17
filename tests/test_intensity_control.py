from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from llm_musical_composer.composition_ir import Composition, Material, Note, Use
from llm_musical_composer.intensity_control import (
    IntensityControlError,
    analyze_intensity_feasibility,
    evaluate_material_intensity_triplet,
    extract_composition_intensity,
    extract_intensity_observables,
    extract_material_intensity,
    find_direction_candidates,
    load_reference_records,
    main,
    resolve_intensity_target,
    resolve_material_intensity_target,
    write_intensity_feasibility,
)
from llm_musical_composer.reference_profile import (
    ReferencePiece,
    extract_reference_profile,
    profile_distances,
)


def _distribution_at(size: int, position: float) -> list[float]:
    scaled = position * (size - 1)
    lower = int(scaled)
    upper = min(size - 1, lower + 1)
    fraction = scaled - lower
    values = [0.0] * size
    values[lower] = 1.0 - fraction
    values[upper] += fraction
    return values


def _record(
    name: str,
    *,
    attack_rate: float,
    polyphony: float,
    velocity: float,
    neighbors: list[str],
) -> dict[str, object]:
    duration_ms = 10_000
    attack_count = round(attack_rate * duration_ms / 1000)
    return {
        "name": name,
        "status": "pass",
        "profile": {
            "status": "pass",
            "note_count": attack_count,
            "attack_count": attack_count,
            "duration_ms": duration_ms,
            "feature_groups": {
                "performance_texture": {
                    "metrics": {
                        "polyphony_duration": {
                            "kind": "distribution",
                            "values": _distribution_at(5, polyphony / 4),
                        },
                        "velocity": {
                            "kind": "distribution",
                            "values": _distribution_at(8, velocity),
                        },
                    }
                }
            },
        },
        "neighborhood": {
            "anchor": name,
            "neighbors": [
                {"name": neighbor, "role": "anchor" if neighbor == name else "neighbor"}
                for neighbor in neighbors
            ],
        },
    }


def _monotone_records() -> list[dict[str, object]]:
    names = [f"piece-{index}.mid" for index in range(7)]
    return [
        _record(
            name,
            attack_rate=1 + index,
            polyphony=0.5 + index * 0.4,
            velocity=0.1 + index * 0.12,
            neighbors=names,
        )
        for index, name in enumerate(names)
    ]


def test_extracts_three_separate_observables() -> None:
    record = _record(
        "anchor.mid",
        attack_rate=3.0,
        polyphony=2.5,
        velocity=0.6,
        neighbors=["anchor.mid", "low.mid", "high.mid"],
    )

    assert extract_intensity_observables(record) == {
        "note_rate_hz": 3.0,
        "attack_rate_hz": 3.0,
        "mean_active_polyphony": 2.5,
        "velocity_level": 0.6,
    }


def test_time_stretch_changes_only_absolute_attack_rate() -> None:
    notes = [
        {"pitch": 60, "onset_ms": 0, "duration_ms": 400, "velocity": 48},
        {"pitch": 64, "onset_ms": 500, "duration_ms": 400, "velocity": 80},
        {"pitch": 67, "onset_ms": 1000, "duration_ms": 400, "velocity": 96},
    ]
    original_piece = ReferencePiece.from_dicts(name="a.mid", notes=notes, pedals=[])
    stretched_piece = ReferencePiece.from_dicts(
        name="a.mid",
        notes=[
            {
                **note,
                "onset_ms": note["onset_ms"] * 2,
                "duration_ms": note["duration_ms"] * 2,
            }
            for note in notes
        ],
        pedals=[],
    )
    original = _record(
        "a.mid", attack_rate=2.15, polyphony=1.0, velocity=0.5, neighbors=["a.mid"] * 3
    )
    stretched = deepcopy(original)
    original["profile"] = extract_reference_profile(original_piece)
    stretched["profile"] = extract_reference_profile(stretched_piece)

    original_values = extract_intensity_observables(original)
    stretched_values = extract_intensity_observables(stretched)

    assert stretched_values["attack_rate_hz"] < original_values["attack_rate_hz"]
    assert stretched_values["note_rate_hz"] < original_values["note_rate_hz"]
    assert stretched_values["mean_active_polyphony"] == pytest.approx(
        original_values["mean_active_polyphony"]
    )
    assert stretched_values["velocity_level"] == pytest.approx(original_values["velocity_level"])
    distances = profile_distances(original["profile"], stretched["profile"])
    assert distances["pitch_harmony"] == 0


@pytest.mark.parametrize(
    ("field", "expected_changed"),
    [
        ("note_count", "note_rate_hz"),
        ("attack_count", "attack_rate_hz"),
        ("polyphony_duration", "mean_active_polyphony"),
        ("velocity", "velocity_level"),
    ],
)
def test_each_synthetic_change_moves_only_its_axis(field: str, expected_changed: str) -> None:
    baseline = _record(
        "a.mid", attack_rate=2.0, polyphony=1.0, velocity=0.3, neighbors=["a.mid"] * 3
    )
    changed = deepcopy(baseline)
    profile = changed["profile"]
    if field == "note_count":
        profile[field] = 30
    elif field == "attack_count":
        profile[field] = 10
    else:
        profile["feature_groups"]["performance_texture"]["metrics"][field]["values"] = (
            _distribution_at(5, 0.75) if field == "polyphony_duration" else _distribution_at(8, 0.8)
        )

    first = extract_intensity_observables(baseline)
    second = extract_intensity_observables(changed)

    assert {key for key in first if first[key] != second[key]} == {expected_changed}


def test_transposition_and_input_order_do_not_change_observables() -> None:
    notes = [
        {"pitch": 60, "onset_ms": 0, "duration_ms": 700, "velocity": 48},
        {"pitch": 64, "onset_ms": 0, "duration_ms": 700, "velocity": 72},
        {"pitch": 67, "onset_ms": 800, "duration_ms": 500, "velocity": 96},
    ]
    first = ReferencePiece.from_dicts(name="a.mid", notes=notes, pedals=[])
    second = ReferencePiece.from_dicts(
        name="a.mid",
        notes=[{**note, "pitch": note["pitch"] + 5} for note in reversed(notes)],
        pedals=[],
    )
    records = []
    for piece in (first, second):
        record = _record("a.mid", attack_rate=1, polyphony=1, velocity=0.5, neighbors=["a.mid"] * 3)
        record["profile"] = extract_reference_profile(piece)
        records.append(record)

    assert extract_intensity_observables(records[0]) == extract_intensity_observables(records[1])


def test_direction_candidates_require_all_three_axes_to_move() -> None:
    anchor = {
        "note_rate_hz": 2.0,
        "attack_rate_hz": 2.0,
        "mean_active_polyphony": 2.0,
        "velocity_level": 0.5,
    }
    candidates = {
        "lower.mid": {
            "note_rate_hz": 1.0,
            "attack_rate_hz": 1.0,
            "mean_active_polyphony": 1.5,
            "velocity_level": 0.4,
        },
        "higher.mid": {
            "note_rate_hz": 3.0,
            "attack_rate_hz": 3.0,
            "mean_active_polyphony": 2.5,
            "velocity_level": 0.7,
        },
        "mixed.mid": {
            "note_rate_hz": 3.0,
            "attack_rate_hz": 3.0,
            "mean_active_polyphony": 2.5,
            "velocity_level": 0.4,
        },
        "equal.mid": dict(anchor),
    }

    result = find_direction_candidates(anchor, candidates)

    assert [item["name"] for item in result["lower"]] == ["lower.mid"]
    assert [item["name"] for item in result["higher"]] == ["higher.mid"]
    assert result["mixed_or_equal_count"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(status="failed"),
        lambda value: value["profile"].update(attack_count=0),
        lambda value: value["profile"]["feature_groups"]["performance_texture"]["metrics"][
            "velocity"
        ].update(values=[1.0, 0.0]),
        lambda value: value["profile"]["feature_groups"]["performance_texture"]["metrics"][
            "polyphony_duration"
        ].update(values=[float("nan"), 0, 0, 0, 1]),
        lambda value: value["neighborhood"].update(anchor="other.mid"),
    ],
)
def test_invalid_records_are_not_converted_to_normal_values(mutation) -> None:
    record = _record(
        "a.mid", attack_rate=2, polyphony=1, velocity=0.5, neighbors=["a.mid", "b.mid", "c.mid"]
    )
    mutation(record)

    with pytest.raises(IntensityControlError):
        extract_intensity_observables(record)


def test_feasibility_uses_default_and_two_quartile_representatives() -> None:
    result = analyze_intensity_feasibility(_monotone_records(), default_reference="piece-3.mid")

    assert result["status"] == "pass"
    assert result["source_count"] == 7
    assert result["representative_anchors"] == [
        "piece-3.mid",
        "piece-1.mid",
        "piece-5.mid",
    ]
    assert all(item["lower_count"] > 0 for item in result["anchor_controls"])
    assert all(item["higher_count"] > 0 for item in result["anchor_controls"])
    assert result["publication_state"] == "not_published"
    assert [item["value"] for item in result["target_resolution"]["piece-3.mid"]] == [
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
    ]
    assert all(
        item["status"] == "reachable"
        for values in result["target_resolution"].values()
        for item in values
    )


def test_feasibility_fails_when_one_axis_conflicts() -> None:
    records = _monotone_records()
    for index, record in enumerate(records):
        velocity = 0.82 - index * 0.12
        record["profile"]["feature_groups"]["performance_texture"]["metrics"]["velocity"][
            "values"
        ] = _distribution_at(8, velocity)

    result = analyze_intensity_feasibility(records, default_reference="piece-3.mid")

    assert result["status"] == "fail"
    assert result["publication_state"] == "not_published"


def test_two_driver_hypothesis_can_pass_when_polyphony_moves_independently() -> None:
    records = _monotone_records()
    for index, record in enumerate(records):
        polyphony = 2.9 - index * 0.35
        record["profile"]["feature_groups"]["performance_texture"]["metrics"]["polyphony_duration"][
            "values"
        ] = _distribution_at(5, polyphony / 4)

    result = analyze_intensity_feasibility(records, default_reference="piece-3.mid")

    assert result["status"] == "fail"
    assert result["alternative_hypotheses"]["note_attack_velocity_drivers"]["status"] == "pass"
    assert result["recommended_hypothesis"] == "note_attack_velocity_drivers"
    assert (
        result["alternative_hypotheses"]["note_attack_velocity_drivers"]["held_observable"]
        == "mean_active_polyphony"
    )


def test_input_order_does_not_change_summary_or_manifest(tmp_path: Path) -> None:
    records = _monotone_records()
    first = write_intensity_feasibility(
        records,
        default_reference="piece-3.mid",
        output_dir=tmp_path / "first",
        input_sha256="a" * 64,
        implementation_sha256="b" * 64,
    )
    second = write_intensity_feasibility(
        list(reversed(records)),
        default_reference="piece-3.mid",
        output_dir=tmp_path / "second",
        input_sha256="a" * 64,
        implementation_sha256="b" * 64,
    )

    assert first == second
    assert json.loads((tmp_path / "first" / "summary.json").read_text(encoding="utf-8")) == first
    assert (tmp_path / "first" / "summary.json").read_bytes() == (
        tmp_path / "second" / "summary.json"
    ).read_bytes()
    assert (tmp_path / "first" / "manifest.json").read_bytes() == (
        tmp_path / "second" / "manifest.json"
    ).read_bytes()


def test_excluded_noise_files_are_rejected() -> None:
    records = _monotone_records()
    records[0]["name"] = "rut.mid"

    with pytest.raises(IntensityControlError, match="excluded"):
        analyze_intensity_feasibility(records, default_reference="piece-3.mid")


def test_load_reference_records_rejects_invalid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "files.jsonl"
    path.write_text('{"name":"ok.mid"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(IntensityControlError, match="line 2"):
        load_reference_records(path)


@pytest.mark.parametrize(
    "content",
    ["", "\n", "[]\n"],
)
def test_load_reference_records_rejects_unusable_files(tmp_path: Path, content: str) -> None:
    path = tmp_path / "files.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(IntensityControlError):
        load_reference_records(path)


def test_load_reference_records_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IntensityControlError, match="unable to read"):
        load_reference_records(tmp_path / "missing.jsonl")


def test_cli_writes_feasibility_outputs(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "files.jsonl"
    input_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in _monotone_records()),
        encoding="utf-8",
    )
    reference_summary = tmp_path / "reference-summary.json"
    reference_summary.write_text(
        json.dumps(
            {
                "status": "pass",
                "default_reference": {"selected": {"kind": "real_medoid", "name": "piece-3.mid"}},
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--input-jsonl",
                str(input_path),
                "--reference-summary",
                str(reference_summary),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "representative_anchors": ["piece-3.mid", "piece-1.mid", "piece-5.mid"],
        "source_count": 7,
        "status": "pass",
    }
    assert (tmp_path / "out" / "manifest.json").is_file()


@pytest.mark.parametrize(
    ("value", "attack_rate", "velocity"),
    [
        (-1.0, 1.0, 0.1),
        (-0.5, 2.5, 0.28),
        (0.0, 4.0, 0.46),
        (0.5, 5.5, 0.64),
        (1.0, 7.0, 0.82),
    ],
)
def test_resolve_intensity_target_is_piecewise_linear_and_monotone(
    value: float, attack_rate: float, velocity: float
) -> None:
    result = resolve_intensity_target(
        _monotone_records(), reference_name="piece-3.mid", value=value, max_notes=2000
    )

    assert result["status"] == "reachable"
    assert result["value"] == value
    assert result["targets"]["attack_rate_hz"] == pytest.approx(attack_rate)
    assert result["targets"]["note_rate_hz"] == pytest.approx(attack_rate)
    assert result["targets"]["velocity_level"] == pytest.approx(velocity)
    assert result["holds"]["mean_active_polyphony"] == {
        "minimum": 0.5,
        "maximum": 2.9,
    }
    assert result["evidence"]["lower_reference"] == "piece-0.mid"
    assert result["evidence"]["higher_reference"] == "piece-6.mid"


def test_resolver_prefers_balanced_progress_over_one_extreme_axis() -> None:
    records = _monotone_records()
    names = [record["name"] for record in records]
    records.append(
        _record(
            "unbalanced.mid",
            attack_rate=0.1,
            polyphony=1.5,
            velocity=0.44,
            neighbors=[*names, "unbalanced.mid"],
        )
    )
    for record in records:
        if record["name"] == "piece-3.mid":
            record["neighborhood"]["neighbors"].append(
                {"name": "unbalanced.mid", "role": "neighbor"}
            )

    result = resolve_intensity_target(records, reference_name="piece-3.mid", value=-1)

    assert result["evidence"]["lower_reference"] == "piece-0.mid"


def test_resolver_reports_unreachable_without_both_directions() -> None:
    records = _monotone_records()
    for index, record in enumerate(records[:3]):
        record["profile"]["feature_groups"]["performance_texture"]["metrics"]["velocity"][
            "values"
        ] = _distribution_at(8, 0.8 - index * 0.2)
    anchor = next(record for record in records if record["name"] == "piece-0.mid")
    anchor["neighborhood"]["neighbors"] = [
        {"name": record["name"], "role": "anchor" if record is anchor else "neighbor"}
        for record in records[:3]
    ]

    result = resolve_intensity_target(records, reference_name="piece-0.mid", value=0.5)

    assert result == {
        "schema_version": 1,
        "status": "unreachable",
        "reference": "piece-0.mid",
        "value": 0.5,
        "reason": "local neighborhood has no strictly lower two-driver endpoint",
    }


def test_resolver_enforces_preset_note_capacity_before_generation() -> None:
    records = _monotone_records()
    high_anchor = next(record for record in records if record["name"] == "piece-5.mid")
    high_anchor["neighborhood"]["neighbors"] = [
        {"name": record["name"], "role": "anchor" if record is high_anchor else "neighbor"}
        for record in records[4:]
    ]

    clipped = resolve_intensity_target(records, reference_name="piece-3.mid", value=1.0)
    unreachable = resolve_intensity_target(records, reference_name="piece-5.mid", value=0.0)

    assert clipped["targets"]["attack_rate_hz"] == pytest.approx(950 / 180)
    assert clipped["evidence"]["note_rate_clipped_to_preset"] is True
    assert unreachable == {
        "schema_version": 1,
        "status": "unreachable",
        "reference": "piece-5.mid",
        "value": 0.0,
        "reason": "reference center exceeds preset note capacity",
        "preset_note_capacity_hz": round(950 / 180, 8),
    }


def test_feasibility_chooses_reachable_neighbor_when_default_is_unreachable() -> None:
    records = _monotone_records()
    default = next(record for record in records if record["name"] == "piece-5.mid")
    default["neighborhood"]["neighbors"] = [
        {
            "name": record["name"],
            "role": "anchor" if record is default else "neighbor",
            "group_ranks": {
                "performance_texture": index,
                "rhythm_time": index,
                "pitch_harmony": index,
            },
        }
        for index, record in enumerate(records[3:])
    ]

    result = analyze_intensity_feasibility(records, default_reference="piece-5.mid")

    assert result["preset_reachability"]["control_default_reference"] == "piece-3.mid"
    assert result["preset_reachability"]["control_default_selection_basis"] == (
        "nearest_preset_reachable_neighbor_of_reference_profile_medoid"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("profile"),
        lambda value: value["profile"].update(status="failed"),
        lambda value: value["profile"].update(note_count=1, attack_count=2),
        lambda value: value["profile"]["feature_groups"]["performance_texture"]["metrics"][
            "velocity"
        ].update(kind="scalar"),
        lambda value: value["profile"]["feature_groups"]["performance_texture"]["metrics"][
            "velocity"
        ].update(values=[0.2] * 8),
        lambda value: value["profile"]["feature_groups"]["performance_texture"]["metrics"][
            "velocity"
        ].update(values=[True, 0, 0, 0, 0, 0, 0, 0]),
    ],
)
def test_more_invalid_profile_shapes_are_rejected(mutation) -> None:
    record = _record("a.mid", attack_rate=2, polyphony=1, velocity=0.5, neighbors=["a.mid"] * 3)
    mutation(record)

    with pytest.raises(IntensityControlError):
        extract_intensity_observables(record)


@pytest.mark.parametrize("axes", [(), ("unknown",), ("note_rate_hz", "note_rate_hz")])
def test_direction_axes_must_be_a_unique_observable_subset(axes) -> None:
    anchor = {
        "note_rate_hz": 1,
        "attack_rate_hz": 1,
        "mean_active_polyphony": 1,
        "velocity_level": 0.5,
    }
    with pytest.raises(IntensityControlError):
        find_direction_candidates(anchor, {}, direction_axes=axes)


@pytest.mark.parametrize(
    "anchor",
    [
        {"note_rate_hz": 1},
        {
            "note_rate_hz": True,
            "attack_rate_hz": 1,
            "mean_active_polyphony": 1,
            "velocity_level": 0.5,
        },
        {
            "note_rate_hz": -1,
            "attack_rate_hz": 1,
            "mean_active_polyphony": 1,
            "velocity_level": 0.5,
        },
    ],
)
def test_direction_candidates_reject_invalid_observables(anchor) -> None:
    with pytest.raises(IntensityControlError):
        find_direction_candidates(anchor, {})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_seconds": 0},
        {"duration_seconds": float("nan")},
        {"max_notes": 0},
        {"max_notes": True},
    ],
)
def test_resolver_rejects_invalid_preset_limits(kwargs) -> None:
    with pytest.raises(IntensityControlError):
        resolve_intensity_target(
            _monotone_records(), reference_name="piece-3.mid", value=0, **kwargs
        )


def test_resolver_rejects_unknown_reference_and_neighbor() -> None:
    records = _monotone_records()
    with pytest.raises(IntensityControlError, match="reference is unavailable"):
        resolve_intensity_target(records, reference_name="missing.mid", value=0)
    records[0]["neighborhood"]["neighbors"][1]["name"] = "missing.mid"
    with pytest.raises(IntensityControlError, match="unknown neighborhood"):
        resolve_intensity_target(records, reference_name="piece-0.mid", value=0)


def _intensity_material(
    material_id: str, *, attack_count: int, chord_size: int = 2, velocity: int = 80
) -> Material:
    duration_ms = 10_000
    step_ms = duration_ms // attack_count
    note_duration_ms = max(1, round(step_ms * 0.4))
    notes = tuple(
        Note(
            f"{material_id}-{attack}-{voice}",
            attack * step_ms,
            note_duration_ms,
            48 + voice * 12,
            velocity,
            "lower" if voice == 0 else "upper",
        )
        for attack in range(attack_count)
        for voice in range(chord_size)
    )
    return Material(material_id, duration_ms, notes)


def test_material_intensity_uses_the_declared_material_duration() -> None:
    material = _intensity_material("A", attack_count=10)

    result = extract_material_intensity(material)

    assert result["duration_ms"] == 10_000
    assert result["note_count"] == 20
    assert result["attack_count"] == 10
    assert result["observables"]["note_rate_hz"] == 2.0
    assert result["observables"]["attack_rate_hz"] == 1.0


@pytest.mark.parametrize(
    "material",
    [
        Material("empty", 1_000, ()),
        Material("outside", 1_000, (Note("n", 900, 200, 60, 80, "upper"),)),
    ],
)
def test_material_intensity_rejects_empty_or_out_of_bounds_notes(material: Material) -> None:
    with pytest.raises(IntensityControlError):
        extract_material_intensity(material)


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_material_target_is_relative_to_the_same_material(value: float) -> None:
    material = _intensity_material("A", attack_count=10, velocity=48)
    records = _monotone_records()
    reference = resolve_intensity_target(
        records, reference_name="piece-3.mid", value=value, max_notes=2000
    )
    center = resolve_intensity_target(
        records, reference_name="piece-3.mid", value=0.0, max_notes=2000
    )

    result = resolve_material_intensity_target(
        material,
        records,
        reference_name="piece-3.mid",
        value=value,
        max_notes=2000,
    )

    baseline = extract_material_intensity(material)["observables"]
    assert result["status"] == "reachable"
    assert result["targets"]["note_rate_hz"] == pytest.approx(
        baseline["note_rate_hz"]
        * reference["targets"]["note_rate_hz"]
        / center["targets"]["note_rate_hz"]
    )
    assert result["targets"]["attack_rate_hz"] == pytest.approx(
        baseline["attack_rate_hz"]
        * reference["targets"]["attack_rate_hz"]
        / center["targets"]["attack_rate_hz"]
    )
    assert result["targets"]["velocity_level"] == pytest.approx(
        baseline["velocity_level"]
        + reference["targets"]["velocity_level"]
        - center["targets"]["velocity_level"]
    )
    assert (
        result["holds"]["mean_active_polyphony"]["minimum"]
        <= baseline["mean_active_polyphony"]
        <= result["holds"]["mean_active_polyphony"]["maximum"]
    )


def test_material_target_rejects_an_unreachable_relative_velocity() -> None:
    material = _intensity_material("A", attack_count=10, velocity=127)

    result = resolve_material_intensity_target(
        material,
        _monotone_records(),
        reference_name="piece-3.mid",
        value=1.0,
        max_notes=2000,
    )

    assert result["status"] == "unreachable"
    assert "velocity" in result["reason"]


def test_triplet_requires_meaningful_monotonic_progress_and_polyphony_hold() -> None:
    center = _intensity_material("A", attack_count=10, velocity=48)
    low = _intensity_material("A", attack_count=6, velocity=16)
    high = _intensity_material("A", attack_count=14, velocity=96)
    targets = {
        value: resolve_material_intensity_target(
            center,
            _monotone_records(),
            reference_name="piece-3.mid",
            value=value,
            max_notes=2000,
        )
        for value in (-1.0, 0.0, 1.0)
    }

    report = evaluate_material_intensity_triplet({-1.0: low, 0.0: center, 1.0: high}, targets)

    assert report["status"] == "pass"
    assert report["direction_status"] == "pass"
    assert report["hold_status"] == "pass"


def test_triplet_rejects_tiny_or_chord_only_progress() -> None:
    center = _intensity_material("A", attack_count=10, velocity=48)
    low = _intensity_material("A", attack_count=6, velocity=16)
    chord_only_high = _intensity_material("A", attack_count=10, chord_size=3, velocity=96)
    targets = {
        value: resolve_material_intensity_target(
            center,
            _monotone_records(),
            reference_name="piece-3.mid",
            value=value,
            max_notes=2000,
        )
        for value in (-1.0, 0.0, 1.0)
    }

    report = evaluate_material_intensity_triplet(
        {-1.0: low, 0.0: center, 1.0: chord_only_high}, targets
    )

    assert report["status"] == "fail"
    assert "attack_rate_hz" in report["issues"]


def test_triplet_rejects_progress_that_is_too_small_for_the_resolved_endpoint() -> None:
    center = _intensity_material("A", attack_count=10, velocity=48)
    low = _intensity_material("A", attack_count=6, velocity=16)
    barely_high = _intensity_material("A", attack_count=11, velocity=96)
    targets = {
        value: resolve_material_intensity_target(
            center,
            _monotone_records(),
            reference_name="piece-3.mid",
            value=value,
            max_notes=2000,
        )
        for value in (-1.0, 0.0, 1.0)
    }

    report = evaluate_material_intensity_triplet(
        {-1.0: low, 0.0: center, 1.0: barely_high}, targets
    )

    assert report["status"] == "fail"
    assert "attack_rate_hz" in report["issues"]


def test_triplet_requires_exact_control_keys() -> None:
    material = _intensity_material("A", attack_count=10, velocity=48)
    with pytest.raises(IntensityControlError, match="exactly"):
        evaluate_material_intensity_triplet({0.0: material}, {0.0: {}})


def test_composition_intensity_expands_repeated_material_without_changing_rates() -> None:
    material = Material(
        material_id="A",
        duration_ms=1000,
        notes=(
            Note("n1", 0, 400, 60, 64, "lower"),
            Note("n2", 0, 400, 72, 80, "upper"),
            Note("n3", 500, 500, 74, 96, "upper"),
        ),
    )
    once = Composition(title="once", form=(Use("A"),), materials=(material,))
    twice = Composition(title="twice", form=(Use("A"), Use("A")), materials=(material,))

    first = extract_composition_intensity(once)
    second = extract_composition_intensity(twice)

    assert first["basis"] == "expanded composition body without generated tonic ending"
    assert first["observables"] == second["observables"]
    assert first["expanded_note_count"] == 3
    assert second["expanded_note_count"] == 6


def test_composition_intensity_rejects_empty_body() -> None:
    composition = Composition(
        title="empty",
        form=(Use("A"),),
        materials=(Material("A", 1000, ()),),
    )

    with pytest.raises(IntensityControlError, match="completed note"):
        extract_composition_intensity(composition)


@pytest.mark.parametrize(
    ("input_digest", "implementation_digest"),
    [("short", "b" * 64), ("a" * 64, "not-hex" * 9 + "x")],
)
def test_writer_rejects_invalid_hashes(
    tmp_path: Path, input_digest: str, implementation_digest: str
) -> None:
    with pytest.raises(IntensityControlError, match="sha256"):
        write_intensity_feasibility(
            _monotone_records(),
            default_reference="piece-3.mid",
            output_dir=tmp_path,
            input_sha256=input_digest,
            implementation_sha256=implementation_digest,
        )


def test_cli_rejects_reference_summary_that_has_not_passed(tmp_path: Path) -> None:
    input_path = tmp_path / "files.jsonl"
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in _monotone_records()),
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"status": "fail"}), encoding="utf-8")

    with pytest.raises(IntensityControlError, match="default reference"):
        main(
            [
                "--input-jsonl",
                str(input_path),
                "--reference-summary",
                str(summary_path),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.1, 1.1, True])
def test_resolver_rejects_invalid_control_value(value) -> None:
    with pytest.raises(IntensityControlError):
        resolve_intensity_target(_monotone_records(), reference_name="piece-3.mid", value=value)
