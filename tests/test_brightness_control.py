from __future__ import annotations

import json
import shutil
from itertools import pairwise
from pathlib import Path

import mido
import pytest

from llm_musical_composer.brightness_control import (
    SCHEME_ID,
    audit_score_layout,
    build_brightness_variants,
    generated_brightness_descriptor,
)
from llm_musical_composer.brightness_control_run import (
    EXPECTED_INPUT_HASHES,
    _quality_gate,
    run_brightness_control,
)
from llm_musical_composer.height_control import _new_or_worsened_voice_collisions
from llm_musical_composer.performance_pipeline import render_performance, validate_pipeline
from llm_musical_composer.pipeline_dsl import (
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.recurrence_quality import (
    analyze_foreground_dissonance,
    analyze_material_harmony,
    analyze_transition_connection,
)

ROOT = Path(__file__).parents[1]
INPUT_DIR = ROOT / ".appendix" / "multiscale-calibration-run-v8" / "inputs"
EXPECTED_SMF_HASHES = {
    "low": "8edfc58e1f06a65054f16365d665d3e364956391d8a44ab9db143ca5744b7128",
    "high": "04fec227f64efc21c566e562b12978807d4c79318d5cc697f7acb1fd4dc107f1",
}
EXPECTED_BRIGHTNESS_RAW = {"low": -0.35283035, "high": 0.38398258}


def _inputs():
    return (
        parse_piece_plan((INPUT_DIR / "piece-plan.music.py").read_text(encoding="utf-8")),
        parse_score_spec((INPUT_DIR / "score.music.py").read_text(encoding="utf-8")),
        parse_performance_spec((INPUT_DIR / "performance.music.py").read_text(encoding="utf-8")),
    )


def _non_pitch_note(note):
    return (
        note.event_id,
        note.at_units,
        note.duration_units,
        note.voice,
        note.tie,
        note.articulations,
    )


def _score_non_pitch_shape(score):
    return tuple(
        (
            material.material_id,
            material.length_units,
            material.derived_from,
            material.directions,
            material.foreground_voice,
            tuple(_non_pitch_note(note) for note in material.notes),
            tuple((item.at_units, item.duration_units) for item in material.harmonies),
        )
        for material in score.materials
    )


def _harmony_signature(material):
    return tuple((harmony.root_pitch_class, harmony.quality) for harmony in material.harmonies)


def _upper_line(material):
    return tuple(
        note
        for note in sorted(material.notes, key=lambda item: (item.at_units, item.event_id))
        if note.voice == "upper"
    )


def _boundary_pitch(material, *, last: bool):
    upper = _upper_line(material)
    onset = (max if last else min)(note.at_units for note in upper)
    return max(note.pitch for note in upper if note.at_units == onset)


def test_v8_layout_is_explicit_before_transforming() -> None:
    _, score, _ = _inputs()

    assert audit_score_layout(score) == {
        "four_harmony_material_ids": [
            "a2-theme",
            "a2-theme-prime",
            "a2-contrast",
            "a2-contrast-prime",
            "a2-return",
            "a2-return-prime",
            "a1-theme",
            "a1-theme-prime",
            "a1-contrast",
            "a1-contrast-prime",
            "a1-return",
            "a1-return-prime",
            "b-theme",
            "b-theme-prime",
            "b-contrast",
            "b-contrast-prime",
            "b-return",
            "b-return-prime",
        ],
        "two_harmony_material_ids": ["transition-ab", "transition-ba2"],
        "harmony_free_material_ids": ["ending", "final-tonic"],
    }


def test_variants_are_only_validated_whole_song_endpoints() -> None:
    plan, score, performance = _inputs()
    variants = build_brightness_variants(plan, score, performance)

    assert set(variants) == {"low", "high"}
    assert variants["low"].plan.mode == "minor"
    assert variants["high"].plan.mode == "major"
    assert variants["low"].performance == performance
    assert variants["high"].performance == performance
    assert _score_non_pitch_shape(variants["low"].score) == _score_non_pitch_shape(score)
    assert _score_non_pitch_shape(variants["high"].score) == _score_non_pitch_shape(score)

    low = {material.material_id: material for material in variants["low"].score.materials}
    high = {material.material_id: material for material in variants["high"].score.materials}
    assert _harmony_signature(low["transition-ab"]) == ((0, "major"), (9, "minor"))
    assert _harmony_signature(low["transition-ba2"]) == ((2, "minor"), (2, "minor"))
    assert _harmony_signature(high["transition-ab"]) == ((9, "major"), (2, "major"))
    assert _harmony_signature(high["transition-ba2"]) == ((2, "major"), (2, "major"))


def test_endpoint_transitions_use_safe_declared_chord_paths() -> None:
    plan, score, performance = _inputs()
    variants = build_brightness_variants(plan, score, performance)
    sources = {
        "transition-ab": ("a1-return-prime", "b-theme", "a1", "b"),
        "transition-ba2": ("b-return-prime", "a2-theme", "b", "a2"),
    }

    for variant in variants.values():
        by_id = {material.material_id: material for material in variant.score.materials}
        for transition_id, (source_id, target_id, source_node, target_node) in sources.items():
            transition = by_id[transition_id]
            line = _upper_line(transition)
            pitches = (
                _boundary_pitch(by_id[source_id], last=True),
                *(note.pitch for note in line),
                _boundary_pitch(by_id[target_id], last=False),
            )
            assert all(left != right for left, right in pairwise(pitches))
            assert all(abs(left - right) <= 5 for left, right in pairwise(pitches[1:-1]))
            assert abs(pitches[1] - pitches[0]) <= 2
            assert abs(pitches[-1] - pitches[-2]) <= 2
            assert analyze_transition_connection(
                variant.plan,
                variant.score,
                source_node_id=source_node,
                transition_node_id=transition_id,
                target_node_id=target_node,
                voice="upper",
                minimum_transition_attacks=6,
                maximum_boundary_leap=2,
                maximum_internal_leap=5,
            ).passes
            assert not analyze_foreground_dissonance(variant.score, transition_id).unsupported_tones


def test_endpoint_variants_pass_existing_quality_and_collision_gates() -> None:
    plan, score, performance = _inputs()
    variants = build_brightness_variants(plan, score, performance)
    base_by_id = {material.material_id: material for material in score.materials}

    for variant in variants.values():
        validate_pipeline(variant.plan, variant.score, variant.performance)
        rendered = render_performance(variant.plan, variant.score, variant.performance)
        assert _quality_gate(
            variant.plan,
            variant.score,
            variant.performance,
            rendered,
            score,
        )["passes"]
        for material in variant.score.materials:
            assert not _new_or_worsened_voice_collisions(base_by_id[material.material_id], material)
            if not material.harmonies:
                continue
            harmony = analyze_material_harmony(
                variant.score,
                material.material_id,
                low_pitch_boundary=48,
                minimum_low_spacing_semitones=7,
            )
            assert harmony.passes
            assert harmony.accompaniment_chord_tone_ratio == 1.0
            assert harmony.low_spacing_violations == 0
            assert not analyze_foreground_dissonance(
                variant.score, material.material_id
            ).unsupported_tones


def test_declared_tonic_brightness_moves_in_the_same_direction() -> None:
    plan, score, performance = _inputs()
    variants = build_brightness_variants(plan, score, performance)
    rendered = {
        level: render_performance(item.plan, item.score, item.performance)
        for level, item in variants.items()
    }
    brightness = {
        level: generated_brightness_descriptor(variants[level].plan, rendered[level])
        for level in variants
    }

    for metric in ("major_minor_profile_margin", "modal_degree_balance"):
        assert (
            brightness["low"]["metrics"][metric]["value"]
            < brightness["high"]["metrics"][metric]["value"]
        )
    assert brightness["low"]["tonic_pitch_class"] == 9
    assert brightness["high"]["tonic_pitch_class"] == 9


def test_transform_is_deterministic() -> None:
    plan, score, performance = _inputs()

    first = build_brightness_variants(plan, score, performance)
    second = build_brightness_variants(plan, score, performance)

    assert first == second
    assert SCHEME_ID == "brightness-v2-whole-song-tonal-endpoints"


def test_run_matches_diagnostic_endpoints_and_defers_listening(tmp_path: Path) -> None:
    output = tmp_path / "run"

    first = run_brightness_control(output_dir=output)
    second = run_brightness_control(output_dir=output)

    assert first["status"] == "machine_passed_listening_deferred"
    assert second == first
    assert first["input_hashes"] == EXPECTED_INPUT_HASHES
    assert {item["level"] for item in first["variants"]} == {"low", "high"}
    assert not (output / "listening-mapping.json").exists()
    assert not (output / "listening-response.json").exists()
    assert first["source_diagnostic_hashes"]["script"] == (
        "81352f5cd23ba759576e658258fab2bc99e3cd2b340589d3c4f7f9c5c6858e5a"
    )

    by_level = {item["level"]: item for item in first["variants"]}
    for level in ("low", "high"):
        item = by_level[level]
        midi_path = output / item["smf"]
        assert midi_path.is_file()
        assert mido.MidiFile(midi_path).length == pytest.approx(180.0, abs=0.001)
        assert item["smf_sha256"] == EXPECTED_SMF_HASHES[level]
        assert item["achieved_raw"] == EXPECTED_BRIGHTNESS_RAW[level]
        assert item["quality_gate"]["passes"]
        assert item["requested"] == (-1 if level == "low" else 1)
        assert item["target_raw"] != item["achieved_raw"]
        assert item["raw_error"] == pytest.approx(item["achieved_raw"] - item["target_raw"])
        assert set(item["covariates"]) == {"高さ", "重なり", "発音頻度"}
    assert by_level["low"]["achieved_normalized"] == pytest.approx(-0.9859243419)
    assert by_level["high"]["achieved_normalized"] == pytest.approx(0.8521555599)
    assert all(
        by_level["low"]["sections"][section] < by_level["high"]["sections"][section]
        for section in ("a1", "b", "a2")
    )


def test_run_stops_before_output_when_fixed_input_hash_differs(tmp_path: Path) -> None:
    copied_inputs = tmp_path / "inputs"
    shutil.copytree(INPUT_DIR, copied_inputs)
    with (copied_inputs / "score.music.py").open("a", encoding="utf-8") as output:
        output.write("\n")

    with pytest.raises(ValueError, match="input hash mismatch"):
        run_brightness_control(output_dir=tmp_path / "run", input_dir=copied_inputs)
    assert not (tmp_path / "run" / "variants").exists()


def test_result_json_is_structured_and_matches_returned_value(tmp_path: Path) -> None:
    output = tmp_path / "run"
    result = run_brightness_control(output_dir=output)

    stored = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert stored == result
    assert stored["reference_baseline"]["record_count"] == 232
    assert stored["middle_status"] == "not_implemented_pending_endpoint_listening"
