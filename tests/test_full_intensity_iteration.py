from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import llm_musical_composer.full_intensity_iteration as full_intensity
from llm_musical_composer.composition_ir import (
    Composition,
    Material,
    Note,
    Part,
    Pedal,
    TonicEnding,
    Use,
)
from llm_musical_composer.full_intensity_iteration import (
    FullIntensityIterationError,
    build_center_material_targets,
    evaluate_energy_order,
    evaluate_full_candidate_quality,
    evaluate_full_intensity_distribution,
    evaluate_full_intensity_triplet,
    generate_deterministic_intensity_candidate,
    main,
    run_full_intensity_iteration,
)
from llm_musical_composer.long_form_generation import LongFormGenerationError
from llm_musical_composer.music_dsl import DslError


def _material(material_id: str, *, note_count: int, velocity: int) -> Material:
    notes = tuple(
        Note(
            f"{material_id}-n{index}",
            index * 200,
            300,
            48 + index % 5,
            velocity,
            "lower" if index % 3 == 0 else "upper",
        )
        for index in range(note_count)
    )
    return Material(material_id, 5_000, notes)


def _composition(
    *,
    low_notes: int = 10,
    high_notes: int = 20,
    low_velocity: int = 64,
    high_velocity: int = 96,
) -> Composition:
    low = _material("A", note_count=low_notes, velocity=low_velocity)
    high = _material("B", note_count=high_notes, velocity=high_velocity)
    return Composition(
        title="test",
        form=(Use("A", role="opening", energy=1), Use("B", role="climax", energy=5)),
        materials=(low, high),
        tonal_center=0,
        mode="major",
        ending=TonicEnding(4_000),
        parts=(
            Part("P1", "opening", 1, 0, 1),
            Part("P2", "climax", 5, 1, 2),
        ),
    )


def test_center_targets_scale_each_material_without_flattening_energy() -> None:
    composition = _composition()
    center = {
        "status": "reachable",
        "value": 0.0,
        "targets": {
            "note_rate_hz": 2.4,
            "attack_rate_hz": 2.4,
            "velocity_level": 0.6,
        },
        "holds": {"mean_active_polyphony": {"minimum": 0.1, "maximum": 2.0}},
    }

    targets = build_center_material_targets(composition, center)

    assert targets["A"]["target_counts"]["note_count"] < targets["B"]["target_counts"]["note_count"]
    assert targets["A"]["targets"]["velocity_level"] < targets["B"]["targets"]["velocity_level"]
    assert targets["A"]["value"] == 0.0
    assert targets["B"]["status"] == "reachable"


def test_center_targets_reject_an_unreachable_or_wrong_value() -> None:
    with pytest.raises(FullIntensityIterationError, match="center target"):
        build_center_material_targets(_composition(), {"status": "unreachable", "value": 0.0})
    with pytest.raises(FullIntensityIterationError, match="center target"):
        build_center_material_targets(
            _composition(),
            {
                "status": "reachable",
                "value": 1.0,
                "targets": {},
                "holds": {},
            },
        )


@pytest.mark.parametrize(
    "center, message",
    [
        (
            {
                "status": "reachable",
                "value": 0.0,
                "targets": {"note_rate_hz": True},
                "holds": {},
            },
            "must be finite",
        ),
        ({"status": "reachable", "value": 0.0, "targets": {}, "holds": {}}, "invalid"),
    ],
)
def test_center_targets_reject_invalid_numeric_contracts(center: dict, message: str) -> None:
    with pytest.raises(FullIntensityIterationError, match=message):
        build_center_material_targets(_composition(), center)


def test_numeric_helpers_reject_non_finite_values_and_hold_equal_targets() -> None:
    with pytest.raises(FullIntensityIterationError, match="must be finite"):
        full_intensity._finite_number(float("nan"), "value")
    assert full_intensity._relative_acceptance(0.5, 0.5) == {
        "minimum": 0.5,
        "maximum": 0.5,
    }


@pytest.mark.parametrize("axis", ["note_rate_hz", "mean_active_polyphony"])
def test_center_targets_reject_nonpositive_baseline_axes(
    axis: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    observables = {
        "note_rate_hz": 1.0,
        "attack_rate_hz": 1.0,
        "velocity_level": 0.5,
        "mean_active_polyphony": 1.0,
    }
    observables[axis] = 0.0
    monkeypatch.setattr(
        full_intensity,
        "extract_composition_intensity",
        lambda composition: {"observables": observables},
    )
    center = {
        "status": "reachable",
        "value": 0.0,
        "targets": {
            "note_rate_hz": 1.0,
            "attack_rate_hz": 1.0,
            "velocity_level": 0.5,
        },
        "holds": {"mean_active_polyphony": {"minimum": 0.5, "maximum": 1.5}},
    }

    with pytest.raises(FullIntensityIterationError, match="must be positive"):
        build_center_material_targets(_composition(), center)


def test_count_allocator_rejects_unrepresentable_totals() -> None:
    with pytest.raises(FullIntensityIterationError, match="not representable"):
        full_intensity._allocate_expanded_counts({"A": 1.0}, {"A": 2}, desired_total=3, minimum=1)


def test_endpoint_targets_validate_value_and_reachability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(FullIntensityIterationError, match="-1 or 1"):
        full_intensity.build_endpoint_material_targets(
            _composition(), [], reference_name="ref.mid", value=0.0
        )
    monkeypatch.setattr(
        full_intensity,
        "resolve_intensity_target",
        lambda *args, **kwargs: {"status": "unreachable"},
    )
    with pytest.raises(FullIntensityIterationError, match="unreachable"):
        full_intensity.build_endpoint_material_targets(
            _composition(), [], reference_name="ref.mid", value=1.0
        )


def test_endpoint_targets_apply_the_requested_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        full_intensity,
        "resolve_intensity_target",
        lambda *args, **kwargs: {"status": "reachable", "value": 1.0},
    )
    monkeypatch.setattr(
        full_intensity,
        "build_center_material_targets",
        lambda *args, **kwargs: {"A": {"value": 0.0, "evidence": {}}},
    )

    result = full_intensity.build_endpoint_material_targets(
        _composition(), [], reference_name="ref.mid", value=1.0
    )

    assert result["A"]["value"] == 1.0
    assert result["A"]["evidence"]["basis"] == "global endpoint allocated to material baseline"


def test_full_triplet_rejects_a_change_that_exists_in_only_one_short_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compositions = {
        -1.0: _composition(low_notes=8),
        0.0: _composition(low_notes=10),
        1.0: _composition(low_notes=12),
    }
    reports = {
        id(compositions[-1.0]): {
            "status": "pass",
            "observables": {
                "note_rate_hz": 2.8,
                "attack_rate_hz": 2.8,
                "velocity_level": 0.5,
                "mean_active_polyphony": 1.0,
            },
        },
        id(compositions[0.0]): {
            "status": "pass",
            "observables": {
                "note_rate_hz": 3.0,
                "attack_rate_hz": 3.0,
                "velocity_level": 0.5,
                "mean_active_polyphony": 1.0,
            },
        },
        id(compositions[1.0]): {
            "status": "pass",
            "observables": {
                "note_rate_hz": 3.2,
                "attack_rate_hz": 3.2,
                "velocity_level": 0.5,
                "mean_active_polyphony": 1.0,
            },
        },
    }
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.extract_composition_intensity",
        lambda composition: reports[id(composition)],
    )
    targets = {
        value: {
            "status": "reachable",
            "value": value,
            "targets": {
                "note_rate_hz": 2.0 + value,
                "attack_rate_hz": 2.0 + value,
                "velocity_level": 0.5 + value * 0.2,
            },
            "holds": {"mean_active_polyphony": {"minimum": 0.5, "maximum": 1.5}},
        }
        for value in (-1.0, 0.0, 1.0)
    }

    result = evaluate_full_intensity_triplet(compositions, targets)

    assert result["status"] == "fail"
    assert "velocity_level" in result["issues"]
    assert "note_rate_hz" in result["issues"]


def test_distribution_rejects_a_difference_confined_to_one_half() -> None:
    baseline = _composition()
    low = replace(
        baseline,
        materials=(
            replace(baseline.materials[0], notes=baseline.materials[0].notes[:-1]),
            baseline.materials[1],
        ),
    )
    high = baseline

    result = evaluate_full_intensity_distribution(
        {-1.0: low, 0.0: baseline, 1.0: high},
        window_ms=5_000,
    )

    assert result["status"] == "fail"
    assert result["changed_window_count"] == 1
    assert result["window_count"] == 2


def test_distribution_rejects_a_nonpositive_window() -> None:
    with pytest.raises(FullIntensityIterationError, match="must be positive"):
        evaluate_full_intensity_distribution(
            {-1.0: _composition(), 0.0: _composition(), 1.0: _composition()},
            window_ms=0,
        )


def test_full_triplet_validates_exact_keys_and_targets() -> None:
    compositions = {-1.0: _composition(), 0.0: _composition()}
    with pytest.raises(FullIntensityIterationError, match="exactly"):
        evaluate_full_intensity_triplet(compositions, {})
    complete = {value: _composition() for value in (-1.0, 0.0, 1.0)}
    targets = {value: _global_target(value) for value in (-1.0, 0.0, 1.0)}
    targets[0.0] = {**targets[0.0], "status": "unreachable"}
    with pytest.raises(FullIntensityIterationError, match="target is invalid"):
        evaluate_full_intensity_triplet(complete, targets)


def test_triplet_target_axes_must_be_monotone() -> None:
    targets = {value: _global_target(value) for value in (-1.0, 0.0, 1.0)}
    targets[1.0]["targets"]["note_rate_hz"] = targets[0.0]["targets"]["note_rate_hz"]
    with pytest.raises(FullIntensityIterationError, match="not monotone"):
        full_intensity._triplet_acceptance(targets, "note_rate_hz", 0.0)


def test_single_intensity_reports_axis_and_polyphony_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = {value: _global_target(value) for value in (-1.0, 0.0, 1.0)}
    monkeypatch.setattr(
        full_intensity,
        "extract_composition_intensity",
        lambda composition: {
            "status": "pass",
            "observables": {
                "note_rate_hz": 99.0,
                "attack_rate_hz": 99.0,
                "velocity_level": 0.99,
                "mean_active_polyphony": 99.0,
            },
        },
    )
    result = full_intensity.evaluate_single_full_intensity(_composition(), targets, value=0.0)
    assert result["status"] == "fail"
    assert set(result["issues"]) == {
        "note_rate_hz",
        "attack_rate_hz",
        "velocity_level",
        "mean_active_polyphony",
    }
    with pytest.raises(FullIntensityIterationError, match="value is invalid"):
        full_intensity.evaluate_single_full_intensity(_composition(), targets, value=0.5)


def test_energy_order_requires_two_real_activity_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    composition = _composition()
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.evaluate_long_form_structure",
        lambda candidate: {
            "status": "pass",
            "parts": [
                {
                    "part_id": "P1",
                    "energy": 1,
                    "features": {
                        "note_density": 4.0,
                        "velocity_median": 90.0,
                        "polyphony_mean": 2.0,
                    },
                },
                {
                    "part_id": "P2",
                    "energy": 5,
                    "features": {
                        "note_density": 3.0,
                        "velocity_median": 100.0,
                        "polyphony_mean": 1.0,
                    },
                },
            ],
        },
    )

    result = evaluate_energy_order(composition)

    assert result["status"] == "fail"
    assert result["violations"][0]["lower_part"] == "P1"


def test_energy_order_passes_when_two_axes_follow_declared_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composition = _composition()
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.evaluate_long_form_structure",
        lambda candidate: {
            "status": "pass",
            "parts": [
                {
                    "part_id": "P1",
                    "energy": 1,
                    "features": {
                        "note_density": 2.0,
                        "velocity_median": 80.0,
                        "polyphony_mean": 2.0,
                    },
                },
                {
                    "part_id": "P2",
                    "energy": 5,
                    "features": {
                        "note_density": 4.0,
                        "velocity_median": 100.0,
                        "polyphony_mean": 1.0,
                    },
                },
            ],
        },
    )

    assert evaluate_energy_order(composition)["status"] == "pass"


def test_energy_order_is_unavailable_without_part_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        full_intensity,
        "evaluate_long_form_structure",
        lambda candidate: {"status": "unable_to_investigate"},
    )
    result = evaluate_energy_order(_composition())
    assert result["status"] == "unable_to_investigate"
    assert result["violations"] == []


@pytest.mark.parametrize(
    "revised, message",
    [
        (replace(_material("A", note_count=3, velocity=64), material_id="X"), "ID changed"),
        (replace(_material("A", note_count=3, velocity=64), duration_ms=4_000), "duration changed"),
        (
            replace(_material("A", note_count=3, velocity=64), derived_from="B"),
            "derivation changed",
        ),
        (
            replace(
                _material("A", note_count=3, velocity=64),
                pedals=(Pedal("p", 0, 127),),
            ),
            "pedals changed",
        ),
    ],
)
def test_material_revision_freezes_metadata(revised: Material, message: str) -> None:
    with pytest.raises(FullIntensityIterationError, match=message):
        full_intensity._validate_material_revision(
            _material("A", note_count=3, velocity=64), revised
        )


def test_material_revision_freezes_pitch_vocabulary_and_range() -> None:
    current = _material("A", note_count=3, velocity=64)
    unsupported = replace(current, notes=(replace(current.notes[0], pitch=55), *current.notes[1:]))
    with pytest.raises(FullIntensityIterationError, match="unsupported pitch-class"):
        full_intensity._validate_material_revision(current, unsupported)
    outside = replace(current, notes=(replace(current.notes[0], pitch=60), *current.notes[1:]))
    with pytest.raises(FullIntensityIterationError, match="allowed pitch range"):
        full_intensity._validate_material_revision(current, outside)


def test_material_revision_rejects_a_new_upper_delay_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _material("A", note_count=3, velocity=64)
    revised = replace(current, notes=tuple(reversed(current.notes)))
    monkeypatch.setattr(
        full_intensity,
        "evaluate_upper_delay_artifact",
        lambda notes: (
            {"status": "pass", "issue": None}
            if notes == current.notes
            else {"status": "fail", "issue": "late upper voice"}
        ),
    )
    with pytest.raises(FullIntensityIterationError, match="late upper voice"):
        full_intensity._validate_material_revision(current, revised)


def _same_count_targets(composition: Composition, *, value: float) -> dict[str, dict]:
    return {
        material.material_id: {
            "value": value,
            "target_counts": {
                "note_count": len(material.notes),
                "attack_count": len({note.at_ms for note in material.notes}),
            },
            "targets": {"velocity_level": 0.5},
        }
        for material in composition.materials
    }


@pytest.mark.parametrize("value", [-1.0, 0.0])
def test_deterministic_candidate_applies_the_transform_and_round_trips(
    value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    composition = _composition()
    captured: dict[str, Composition] = {}

    def fake_source(candidate: Composition) -> str:
        captured["candidate"] = candidate
        return "composition"

    monkeypatch.setattr(full_intensity, "composition_to_source", fake_source)
    monkeypatch.setattr(
        full_intensity,
        "parse_composition",
        lambda *args, **kwargs: captured["candidate"],
    )

    result = generate_deterministic_intensity_candidate(
        composition,
        material_targets=_same_count_targets(composition, value=value),
    )

    assert result == captured["candidate"]
    assert set(result.material_by_id) == set(composition.material_by_id)


def test_deterministic_candidate_rejects_invalid_targets_and_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composition = _composition()
    with pytest.raises(FullIntensityIterationError, match="do not match"):
        generate_deterministic_intensity_candidate(composition, material_targets={"A": {}})

    targets = _same_count_targets(composition, value=0.0)
    targets["A"] = {"value": 0.0}
    with pytest.raises(FullIntensityIterationError, match="transform failed for A"):
        generate_deterministic_intensity_candidate(composition, material_targets=targets)

    targets = _same_count_targets(composition, value=0.0)
    monkeypatch.setattr(
        full_intensity,
        "parse_composition",
        lambda *args, **kwargs: (_ for _ in ()).throw(DslError("bad assembly")),
    )
    with pytest.raises(FullIntensityIterationError, match="bad assembly"):
        generate_deterministic_intensity_candidate(composition, material_targets=targets)


def test_deterministic_candidate_rejects_a_changed_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composition = _composition()
    monkeypatch.setattr(full_intensity, "parse_composition", lambda *args, **kwargs: composition)

    with pytest.raises(FullIntensityIterationError, match="changed during round-trip"):
        generate_deterministic_intensity_candidate(
            composition,
            material_targets=_same_count_targets(composition, value=0.0),
        )


def _global_target(value: float) -> dict:
    return {
        "status": "reachable",
        "value": value,
        "targets": {
            "note_rate_hz": 2.0 + value * 0.5,
            "attack_rate_hz": 1.5 + value * 0.4,
            "velocity_level": 0.5 + value * 0.1,
        },
        "holds": {"mean_active_polyphony": {"minimum": 0.1, "maximum": 3.0}},
    }


def test_full_quality_passes_when_all_fixed_contracts_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _composition()
    long = replace(
        baseline,
        materials=tuple(replace(item, duration_ms=88_000) for item in baseline.materials),
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration._validate_material_revision",
        lambda current, revised: None,
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.evaluate_sustain_profile",
        lambda materials: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.evaluate_piano_texture",
        lambda candidate: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.evaluate_variation_contracts",
        lambda candidate: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.evaluate_energy_order",
        lambda candidate: {"status": "pass", "structure": {"status": "pass", "issues": []}},
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration._validate_natural_materials",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.parse_composition",
        lambda *args, **kwargs: long,
    )

    assert evaluate_full_candidate_quality(long, long)["status"] == "pass"


def test_full_quality_reports_fixed_contract_regressions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _composition()
    candidate = replace(
        baseline,
        title="changed",
        tonal_center=1,
        materials=(baseline.materials[0],),
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.evaluate_sustain_profile",
        lambda materials: {"status": "fail", "issues": ["sustain"]},
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.evaluate_piano_texture",
        lambda value: {"status": "fail", "issues": ["texture"]},
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.evaluate_variation_contracts",
        lambda value: {"status": "fail", "issues": ["variation"]},
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.evaluate_energy_order",
        lambda value: {
            "status": "fail",
            "structure": {"status": "fail", "issues": ["structure"]},
        },
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration._validate_natural_materials",
        lambda *args: (_ for _ in ()).throw(LongFormGenerationError("harmony")),
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.parse_composition",
        lambda *args, **kwargs: (_ for _ in ()).throw(DslError("bad source")),
    )

    report = evaluate_full_candidate_quality(baseline, candidate, max_notes=1)

    assert report["status"] == "fail"
    assert "structure changed" in report["issues"]
    assert "material set changed" in report["issues"]
    assert "sustain" in report["issues"]
    assert "harmony" in report["issues"]
    assert "composition source is invalid: bad source" in report["issues"]


def test_full_quality_reports_material_revision_duration_and_roundtrip_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _composition()
    candidate = replace(
        baseline,
        materials=(replace(baseline.materials[0], duration_ms=90_000), baseline.materials[1]),
    )
    monkeypatch.setattr(
        full_intensity,
        "evaluate_sustain_profile",
        lambda materials: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        full_intensity,
        "evaluate_piano_texture",
        lambda value: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        full_intensity,
        "evaluate_variation_contracts",
        lambda value: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        full_intensity,
        "evaluate_energy_order",
        lambda value: {"status": "pass", "structure": {"status": "pass", "issues": []}},
    )
    monkeypatch.setattr(full_intensity, "_validate_natural_materials", lambda *args: None)
    monkeypatch.setattr(full_intensity, "parse_composition", lambda *args, **kwargs: baseline)

    report = evaluate_full_candidate_quality(baseline, candidate, max_notes=1)

    assert report["status"] == "fail"
    assert "intensity material duration changed" in report["issues"]
    assert "duration is not 180000 ms" in report["issues"]
    assert "note count exceeds 1" in report["issues"]
    assert "composition source round-trip changed the candidate" in report["issues"]


def test_full_quality_reports_missing_referenced_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _composition()
    candidate = replace(baseline, form=(Use("missing"),))
    monkeypatch.setattr(full_intensity, "_validate_material_revision", lambda *args: None)
    monkeypatch.setattr(
        full_intensity,
        "evaluate_sustain_profile",
        lambda materials: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        full_intensity,
        "evaluate_piano_texture",
        lambda value: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        full_intensity,
        "evaluate_variation_contracts",
        lambda value: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        full_intensity,
        "evaluate_energy_order",
        lambda value: {"status": "pass", "structure": {"status": "pass", "issues": []}},
    )
    monkeypatch.setattr(full_intensity, "_validate_natural_materials", lambda *args: None)
    monkeypatch.setattr(full_intensity, "parse_composition", lambda *args, **kwargs: candidate)

    report = evaluate_full_candidate_quality(baseline, candidate)

    assert "composition references a missing material" in report["issues"]


def _prepare_run_files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "base_source_path": tmp_path / "base.music.py",
        "records_path": tmp_path / "files.jsonl",
        "records_manifest_path": tmp_path / "manifest.json",
        "prompt_path": tmp_path / "prompt.md",
        "schema_path": tmp_path / "schema.json",
    }
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")
    return paths


def _mock_run_dependencies(
    monkeypatch: pytest.MonkeyPatch, composition: Composition, generated: list[float]
) -> None:
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration._verify_records_manifest",
        lambda *args: {},
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.load_reference_records", lambda path: []
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.load_prompt", lambda path: "prompt"
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.parse_composition",
        lambda *args, **kwargs: composition,
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.resolve_intensity_target",
        lambda records, *, value, **kwargs: _global_target(value),
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.build_center_material_targets",
        lambda *args: {item.material_id: {"value": 0.0} for item in composition.materials},
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.build_endpoint_material_targets",
        lambda source, records, *, value, **kwargs: {
            item.material_id: {"value": value} for item in source.materials
        },
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.evaluate_single_full_intensity",
        lambda *args, **kwargs: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.evaluate_full_candidate_quality",
        lambda *args, **kwargs: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.evaluate_full_intensity_triplet",
        lambda *args: {"status": "pass", "issues": []},
    )

    def fake_deterministic(source, *, material_targets):
        value = next(iter(material_targets.values())).get("value", 0.0)
        generated.append(value)
        return source

    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.generate_deterministic_intensity_candidate",
        fake_deterministic,
    )


def _fake_write(run_dir: Path, label: str, candidate: Composition) -> dict[str, Path]:
    source = run_dir / "staged" / f"{label}.music.py"
    midi = run_dir / "staged" / f"{label}.mid"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(label.encode())
    midi.write_bytes(b"MThd")
    return {"source": source, "midi": midi}


def test_run_full_intensity_publishes_only_after_all_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    composition = _composition()
    paths = _prepare_run_files(tmp_path)
    generated: list[float] = []
    _mock_run_dependencies(monkeypatch, composition, generated)
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration._write_staged_candidate", _fake_write
    )

    result = run_full_intensity_iteration(
        **paths,
        run_dir=tmp_path / "run",
        reference_name="ref.mid",
    )

    assert result["status"] == "pass"
    assert generated == [0.0, -1.0, 1.0]
    assert len(result["candidates"]) == 3
    assert all(Path(item["midi_path"]).is_file() for item in result["candidates"].values())


def test_run_full_intensity_stops_after_failed_center(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    composition = _composition()
    paths = _prepare_run_files(tmp_path)
    generated: list[float] = []
    _mock_run_dependencies(monkeypatch, composition, generated)
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.evaluate_single_full_intensity",
        lambda *args, **kwargs: {"status": "fail", "issues": ["note_rate_hz"]},
    )
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration._write_staged_candidate", _fake_write
    )

    result = run_full_intensity_iteration(
        **paths,
        run_dir=tmp_path / "run-fail",
        reference_name="ref.mid",
    )

    assert result["status"] == "fail"
    assert result["stage"] == "center"
    assert generated == [0.0]
    assert not (tmp_path / "run-fail/candidates").exists()


def test_run_full_intensity_reports_an_invalid_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _prepare_run_files(tmp_path)
    monkeypatch.setattr(full_intensity, "_verify_records_manifest", lambda *args: {})
    monkeypatch.setattr(full_intensity, "load_reference_records", lambda path: [])
    monkeypatch.setattr(full_intensity, "load_prompt", lambda path: "prompt")
    monkeypatch.setattr(
        full_intensity,
        "parse_composition",
        lambda *args, **kwargs: (_ for _ in ()).throw(DslError("invalid baseline")),
    )

    with pytest.raises(FullIntensityIterationError, match="invalid baseline"):
        run_full_intensity_iteration(
            **paths,
            run_dir=tmp_path / "invalid-run",
            reference_name="ref.mid",
        )


def test_run_full_intensity_stops_before_generation_for_unreachable_global_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    composition = _composition()
    paths = _prepare_run_files(tmp_path)
    generated: list[float] = []
    _mock_run_dependencies(monkeypatch, composition, generated)
    monkeypatch.setattr(
        full_intensity,
        "resolve_intensity_target",
        lambda records, *, value, **kwargs: {"status": "unreachable", "value": value},
    )

    with pytest.raises(FullIntensityIterationError, match="global targets"):
        run_full_intensity_iteration(
            **paths,
            run_dir=tmp_path / "unreachable-run",
            reference_name="ref.mid",
        )

    assert generated == []


def test_main_returns_status_from_the_full_run(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        "llm_musical_composer.full_intensity_iteration.run_full_intensity_iteration",
        lambda **kwargs: {"status": "pass", "candidates": {}},
    )
    assert main(["--base-source", "base.py", "--run-dir", "run"]) == 0
    assert '"status": "pass"' in capsys.readouterr().out


def test_main_returns_two_for_a_failed_full_run(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        full_intensity,
        "run_full_intensity_iteration",
        lambda **kwargs: {"status": "fail", "candidates": {}},
    )
    assert main(["--base-source", "base.py", "--run-dir", "run"]) == 2
    assert '"status": "fail"' in capsys.readouterr().out
