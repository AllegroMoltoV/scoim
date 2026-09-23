import json
from dataclasses import asdict, replace
from fractions import Fraction
from itertools import accumulate
from pathlib import Path
from typing import cast

import pytest
import rfc8785
from test_scoim_phase3_realization import SequencedRunner, _document, _phase2_ledger, _responses
from test_scoim_phase7_state import _phase7_run

from llm_musical_composer.run_state import sha256_file, sha256_json
from scoim.performance_ir import dataclass_content_sha256
from scoim.phase3_model_contracts import build_harmonic_plan, ordered_leaf_section_ids
from scoim.phase3_realization import Phase3Request, realize_phase3
from scoim.phase3_state import load_complete_phase3_state
from scoim.phase8_bundle import (
    Phase8BundleRequest,
    create_phase8_bundle,
    replay_phase8_bundle,
    verify_phase8_bundle,
)
from scoim.score_ir import piece_plan_from_json
from scoim.score_timing import MAX_SCORE_UNITS, ScoreCapacityError, allocate_score_units
from scoim.validation import IssueCode


@pytest.mark.parametrize("last_weight", [1, 1.01, 1.001, 1.000000000000001])
def test_small_ratio_changes_do_not_change_selected_score_capacity(last_weight: float) -> None:
    document = _document()
    sections = cast(dict[str, dict[str, object]], document["script"]["sections"])
    for section_id, weight in zip(
        ordered_leaf_section_ids(document), (1, 1, 1, last_weight), strict=True
    ):
        sections[section_id]["relative_length"] = weight
    response = {**_responses()[0], "total_score_units": 120}

    plan = build_harmonic_plan(document, response, tonal_center=0, divisions=12)

    assert plan.total_score_units == 120
    assert plan.divisions == 12
    assert list(plan.length_units_by_score_unit.values()) == [30, 30, 30, 30]


@pytest.mark.parametrize("scale", [1, 100, Fraction(1, 100)])
def test_boundary_rounding_preserves_total_and_bounds_error_under_common_scaling(
    scale: int | Fraction,
) -> None:
    relative_lengths = {
        name: float(weight * scale)
        for name, weight in zip(("a", "b", "c", "d"), (1, 2, 4, 5), strict=True)
    }

    lengths, evidence = allocate_score_units(relative_lengths, 37)

    boundaries = [0, *accumulate(lengths.values())]
    ideal_boundaries = [Fraction(0), Fraction(37, 12), Fraction(37, 4), Fraction(259, 12), 37]
    assert boundaries == [0, 3, 9, 22, 37]
    for index, section_id in enumerate(relative_lengths):
        ideal_start, ideal_end = ideal_boundaries[index : index + 2]
        start, end = boundaries[index : index + 2]
        assert abs(start - ideal_start) <= Fraction(1, 2)
        assert abs(end - ideal_end) <= Fraction(1, 2)
        assert abs(lengths[section_id] - (ideal_end - ideal_start)) <= 1
        recorded = dict(item.split("=", 1) for item in evidence[section_id].split("; "))
        assert Fraction(recorded["ideal_start"]) == ideal_start
        assert Fraction(recorded["ideal_end"]) == ideal_end
        assert Fraction(recorded["start_error"]) == start - ideal_start
        assert Fraction(recorded["end_error"]) == end - ideal_end


def test_half_unit_ties_round_cumulative_boundaries_to_even() -> None:
    lengths, _ = allocate_score_units({"a": 1, "b": 1, "c": 1, "d": 1}, 10)

    assert list(accumulate(lengths.values())) == [2, 5, 8, 10]


def test_collapsed_section_is_reported_without_minimum_length_compensation() -> None:
    with pytest.raises(ScoreCapacityError) as caught:
        allocate_score_units({"before": 1, "tiny": 1e-15, "after": 1}, 120)

    assert "section tiny rounds to zero units" in str(caught.value)
    assert "ideal_length=" in str(caught.value)
    assert "total_score_units=120" in str(caught.value)


def test_json_safe_capacity_limit_can_be_saved_without_expanding_the_grid() -> None:
    lengths, evidence = allocate_score_units({"whole": 1}, MAX_SCORE_UNITS)

    saved = rfc8785.dumps({"lengths": lengths, "evidence": evidence})

    assert json.loads(saved)["lengths"] == {"whole": 2**53 - 1}
    with pytest.raises(ScoreCapacityError, match="positive JSON-safe integer"):
        allocate_score_units({"whole": 1}, 2**53)


def test_tiny_decimal_difference_preserves_piece_plan_values_and_hash_after_save() -> None:
    document = _document()
    source_weights = [1, 1, 1, 1.0000000000000007]
    sections = cast(dict[str, dict[str, object]], document["script"]["sections"])
    for section_id, weight in zip(ordered_leaf_section_ids(document), source_weights, strict=True):
        sections[section_id]["relative_length"] = weight
    plan = build_harmonic_plan(
        document,
        {**_responses()[0], "total_score_units": 120},
        tonal_center=0,
        divisions=12,
    )

    saved = rfc8785.dumps(asdict(plan.piece_plan))
    restored = piece_plan_from_json(json.loads(saved))

    assert [node.duration_weight for node in restored.nodes if node.score_unit_id] == source_weights
    assert dataclass_content_sha256(restored) == dataclass_content_sha256(plan.piece_plan)
    assert sum(plan.length_units_by_score_unit.values()) == 120
    rfc8785.dumps(asdict(plan))


@pytest.mark.parametrize(
    "tamper",
    [
        "capacity",
        "evidence",
        pytest.param(0, id="zero-divisions"),
        pytest.param(True, id="boolean-divisions"),
        pytest.param(12.5, id="fractional-divisions"),
        pytest.param(MAX_SCORE_UNITS + 1, id="unsafe-divisions"),
    ],
)
def test_phase3_reload_rejects_invalid_saved_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str | int | float
) -> None:
    monkeypatch.setattr("scoim.phase3_realization.secrets.randbelow", lambda upper: 0)
    document = _document()
    responses = _responses()
    responses[0]["total_score_units"] = 120
    run_dir = tmp_path / "phase3"
    result = realize_phase3(
        Phase3Request(document, _phase2_ledger(document)), SequencedRunner(responses), run_dir
    )
    assert result.realized, result.issues
    state = json.loads((run_dir / "outputs" / "phase3-state.json").read_text("utf-8"))
    ledger = json.loads((run_dir / "outputs" / "projection-ledger.json").read_text("utf-8"))
    loaded = load_complete_phase3_state(document, state, ledger)
    assert loaded.plan.total_score_units == 120

    if tamper == "capacity":
        state["harmonic_plan"]["total_score_units"] = 121
        expected_error = "lengths do not match the saved timing contract"
    elif tamper == "evidence":
        for entries in (
            state["harmonic_plan"]["projection_ledger"],
            state["projection_ledger"],
            ledger,
        ):
            entry = next(
                entry
                for entry in entries
                if entry["verification"] == "quantized_section_boundary"
                and entry["source_id"] == "statement"
            )
            original = entry["evidence"]
            entry["evidence"] = original.replace("end=48;", "end=47;")
            assert entry["evidence"] != original
        expected_error = "quantization evidence does not match the script"
    else:
        state["harmonic_plan"]["divisions"] = tamper
        expected_error = "divisions"

    with pytest.raises(ValueError, match=expected_error):
        load_complete_phase3_state(document, state, ledger)


def test_tiny_ratio_difference_survives_phase3_to_bundle_replay_with_saved_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _document()
    source_weights = [4, 1, 4, 1.0000000000000007]
    sections = cast(dict[str, dict[str, object]], document["script"]["sections"])
    for section_id, weight in zip(ordered_leaf_section_ids(document), source_weights, strict=True):
        sections[section_id]["relative_length"] = weight
    fixture = tmp_path / "tiny-ratio-script.json"
    fixture.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr("test_scoim_phase5_realization._FIXTURE", fixture)

    phase7_dir = _phase7_run(tmp_path)
    saved_plan = json.loads(
        (tmp_path / "phase3" / "outputs" / "piece-plan.json").read_text("utf-8")
    )
    assert [node["duration_weight"] for node in saved_plan["nodes"] if node["score_unit_id"]] == (
        source_weights
    )
    bundle_dir = tmp_path / "bundle"
    request = Phase8BundleRequest(
        phase7_run_dir=phase7_dir,
        phase_run_dirs={f"phase{phase}": tmp_path / f"phase{phase}" for phase in range(3, 7)},
        composition_id="composition-001",
        trial_id="trial-001",
        composition_manifest=json.dumps(
            {
                "bundle_type": "composition",
                "schema_version": 3,
                "target_profile": "solo_piano_3m_v2",
                "composition_id": "composition-001",
            }
        ).encode(),
    )
    created = create_phase8_bundle(request, bundle_dir)

    assert created.created, created.issues
    verification = verify_phase8_bundle(bundle_dir)
    assert verification.valid, verification.issues
    manifest = json.loads((bundle_dir / "manifest.json").read_text("utf-8"))
    assert manifest["piece_plan_sha256"] == sha256_json(saved_plan)
    replay_dir = tmp_path / "replay"
    replayed = replay_phase8_bundle(bundle_dir, replay_dir)
    assert replayed.replayed, replayed.issues
    for artifact in ("score.musicxml", "final.mid"):
        assert sha256_file(replay_dir / artifact) == sha256_file(
            bundle_dir / "artifacts" / artifact
        )

    embedded_phase3 = bundle_dir / "model-runs" / "phase3"
    spec_path = embedded_phase3 / "run-spec.json"
    old_spec = json.loads(spec_path.read_text("utf-8"))
    old_spec["schema_version"] = 2
    old_spec.pop("timing_contract")
    spec_path.write_text(json.dumps(old_spec), encoding="utf-8")
    manifest["files"]["model-runs/phase3/run-spec.json"] = sha256_file(spec_path)
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    mixed_verification = verify_phase8_bundle(bundle_dir)
    assert not mixed_verification.valid
    assert "timing" in mixed_verification.issues[0].message
    mixed_creation = create_phase8_bundle(
        replace(request, phase_run_dirs={**request.phase_run_dirs, "phase3": embedded_phase3}),
        tmp_path / "mixed-bundle",
    )
    assert not mixed_creation.created
    assert "timing" in mixed_creation.issues[0].message
    assert not (tmp_path / "mixed-bundle").exists()


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_zero_unit_overall_plan_requires_capacity_reselection_before_harmony(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repair_succeeds: bool
) -> None:
    monkeypatch.setattr("scoim.phase3_realization.secrets.randbelow", lambda upper: 0)
    document = _document()
    valid_responses = _responses()
    invalid_overall = {**valid_responses[0], "total_score_units": 1}
    responses = (
        [invalid_overall, *valid_responses]
        if repair_succeeds
        else [invalid_overall, invalid_overall]
    )
    runner = SequencedRunner(responses)
    run_dir = tmp_path / "phase3"

    result = realize_phase3(Phase3Request(document, _phase2_ledger(document)), runner, run_dir)

    assert "rounds to zero units" in runner.prompts[1]
    assert "/total_score_units" in runner.prompts[1]
    if repair_succeeds:
        assert result.realized, result.issues
        assert len(runner.prompts) == 6
        state = json.loads((run_dir / "outputs" / "phase3-state.json").read_text("utf-8"))
        assert state["harmonic_plan"]["total_score_units"] == 120
        assert state["harmonic_plan"]["length_units_by_score_unit"] == {
            "score-unit-statement": 48,
            "score-unit-bridge": 12,
            "score-unit-return": 48,
            "score-unit-release": 12,
        }
        summary = json.loads((run_dir / "outputs" / "realization.json").read_text("utf-8"))
        assert summary["repair_count"] == 1
    else:
        assert not result.realized
        assert result.outcome == "unrepresentable"
        assert result.issues[0].code == IssueCode.UNREPRESENTABLE
        assert len(runner.prompts) == 2
        assert not list((run_dir / "attempts").glob("harmony-*"))
        assert not (run_dir / "outputs" / "piece-plan.json").exists()
