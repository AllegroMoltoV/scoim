"""あかるさ制御v2二端点の機械検査とSMF出力を管理する。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from llm_musical_composer.brightness_control import (
    SCHEME_ID,
    build_brightness_variants,
    generated_brightness_descriptor,
)
from llm_musical_composer.control_reference_baseline import extract_control_observables
from llm_musical_composer.height_control import _new_or_worsened_voice_collisions
from llm_musical_composer.performance_pipeline import (
    RenderedPerformance,
    ScoreSpec,
    render_performance,
    render_performance_smf,
    validate_pipeline,
)
from llm_musical_composer.pipeline_dsl import (
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.recurrence_quality import (
    analyze_foreground_dissonance,
    analyze_foreground_variation,
    analyze_material_harmony,
    analyze_rendered_boundary,
    analyze_rendered_foreground_dissonance,
    analyze_rendered_harmony,
    analyze_section_pacing,
    analyze_transition_connection,
)
from llm_musical_composer.reference_profile import ReferencePiece, load_reference_piece
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file

ROOT = Path(__file__).parents[2]
DEFAULT_INPUT_DIR = ROOT / ".appendix" / "multiscale-calibration-run-v8" / "inputs"
DEFAULT_OUTPUT_DIR = ROOT / ".appendix" / "brightness-control-v2-endpoints"
REFERENCE_BASELINE_DIR = ROOT / ".appendix" / "control-reference-baseline-v3"
EXPECTED_INPUT_HASHES = {
    "piece_plan": "62b700f337dd5ccdc800d2ab10e4562b85d377a68a274f4d631b7b7f601488f2",
    "score": "19a295e8611640f292477e5ee09951fa72fb604a7823541e5d0860be040c9275",
    "performance": "8f5ca992934006569ad985df9cdbfec4c981a3f015eb630389a1dc4cc4e0fda2",
}
INPUT_FILES = {
    "piece_plan": "piece-plan.music.py",
    "score": "score.music.py",
    "performance": "performance.music.py",
}
LEVELS = ("low", "high")
EXPECTED_SMF_HASHES = {
    "low": "8edfc58e1f06a65054f16365d665d3e364956391d8a44ab9db143ca5744b7128",
    "high": "04fec227f64efc21c566e562b12978807d4c79318d5cc697f7acb1fd4dc107f1",
}
SOURCE_DIAGNOSTIC_HASHES = {
    "script": "81352f5cd23ba759576e658258fab2bc99e3cd2b340589d3c4f7f9c5c6858e5a",
    "result": "19b9a61eb55ba797fb00e20d5d3745134ada7c2e2d2e403bfcb39109a34e8b6a",
}
TRANSITIONS = (("a1", "transition-ab", "b"), ("b", "transition-ba2", "a2"))
VARIATION_PAIRS = (
    ("a1-theme", "a1-return"),
    ("a1-theme-prime", "a1-return-prime"),
    ("a2-theme", "a2-return"),
    ("a2-theme-prime", "a2-return-prime"),
    ("b-theme", "b-theme-prime"),
    ("b-contrast", "b-contrast-prime"),
    ("b-theme", "b-return"),
    ("b-theme-prime", "b-return-prime"),
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def _input_hashes(input_dir: Path) -> dict[str, str]:
    return {key: sha256_file(input_dir / name) for key, name in INPUT_FILES.items()}


def _load_inputs(input_dir: Path):
    return (
        parse_piece_plan((input_dir / INPUT_FILES["piece_plan"]).read_text(encoding="utf-8")),
        parse_score_spec((input_dir / INPUT_FILES["score"]).read_text(encoding="utf-8")),
        parse_performance_spec(
            (input_dir / INPUT_FILES["performance"]).read_text(encoding="utf-8")
        ),
    )


def _ending_metrics(rendered: RenderedPerformance) -> dict[str, object]:
    notes = tuple(note for note in rendered.notes if note.occurrence_node_id == "a2-2-3")
    attack = max(note.at_ms for note in notes)
    chord = tuple(note for note in notes if note.at_ms == attack)
    release = min(
        pedal.at_ms
        for pedal in rendered.pedals
        if pedal.occurrence_node_id == "a2-2-3" and pedal.value == 0 and pedal.at_ms >= attack
    )
    carryover = sum(
        note.at_ms + note.duration_ms > attack
        for note in rendered.notes
        if note.occurrence_node_id != "a2-2-3"
    )
    return {
        "final_note_count": len(chord),
        "final_pitch_classes": sorted({note.pitch % 12 for note in chord}),
        "minimum_note_duration_ms": min(note.duration_ms for note in chord),
        "pedal_hold_after_attack_ms": release - attack,
        "prior_key_carryover_count": carryover,
    }


def _quality_gate(plan, score: ScoreSpec, performance, rendered, base_score) -> dict[str, object]:
    validate_pipeline(plan, score, performance)
    transitions = tuple(
        analyze_transition_connection(
            plan,
            score,
            source_node_id=source,
            transition_node_id=transition,
            target_node_id=target,
            voice="upper",
            minimum_transition_attacks=6,
            maximum_boundary_leap=2,
            maximum_internal_leap=5,
        )
        for source, transition, target in TRANSITIONS
    )
    boundaries = tuple(
        analyze_rendered_boundary(
            plan,
            rendered,
            transition_node_id=transition,
            target_node_id=target,
            voice="upper",
            maximum_pedal_gap_ms=0,
        )
        for _, transition, target in TRANSITIONS
    )
    variations = tuple(
        analyze_foreground_variation(score, source, target) for source, target in VARIATION_PAIRS
    )
    harmony = tuple(
        analyze_material_harmony(
            score,
            item.material_id,
            low_pitch_boundary=48,
            minimum_low_spacing_semitones=7,
        )
        for item in score.materials
        if item.harmonies
    )
    dissonance = tuple(
        analyze_foreground_dissonance(score, item.material_id)
        for item in score.materials
        if item.harmonies
    )
    rendered_harmony = analyze_rendered_harmony(plan, score, rendered)
    pedal_warning = analyze_rendered_foreground_dissonance(plan, score, rendered)
    pacing = analyze_section_pacing(
        plan,
        score,
        rendered,
        source_node_id="a1",
        target_node_id="a2",
        maximum_target_ratio=0.90,
    )
    pacing_ratio = pacing.target_ms_per_unit / pacing.source_ms_per_unit
    ending = _ending_metrics(rendered)
    expected_tonic = {1, 4, 9} if plan.mode == "major" else {0, 4, 9}
    failures: list[str] = []
    failures += [f"transition:{item.transition_node_id}" for item in transitions if not item.passes]
    failures += [
        f"rendered-boundary:{item.transition_node_id}" for item in boundaries if not item.passes
    ]
    failures += [
        f"foreground-variation:{item.target_material_id}" for item in variations if not item.passes
    ]
    failures += [f"harmony:{item.material_id}" for item in harmony if not item.passes]
    failures += [
        f"unsupported:{item.material_id}"
        for item in dissonance
        if item.status == "assessed" and item.unsupported_tones
    ]
    if not rendered_harmony.passes:
        failures.append("rendered-harmony")
    if not 0.84 <= pacing_ratio <= 0.90:
        failures.append("pacing-range")
    if rendered.duration_ms != 180_000:
        failures.append("duration")
    if sum(map(lambda item: len(item.notes), score.materials)) != sum(
        map(lambda item: len(item.notes), base_score.materials)
    ):
        failures.append("note-count")
    if ending["final_note_count"] != 5:
        failures.append("final-tonic-note-count")
    if set(ending["final_pitch_classes"]) != expected_tonic:
        failures.append("final-tonic-pitches")
    if ending["minimum_note_duration_ms"] < 2_000:
        failures.append("final-tonic-note-duration")
    if ending["pedal_hold_after_attack_ms"] < 2_000:
        failures.append("final-tonic-pedal-duration")
    if ending["prior_key_carryover_count"]:
        failures.append("ending-key-carryover")
    return {
        "passes": not failures,
        "failures": failures,
        "transitions": [asdict(item) for item in transitions],
        "rendered_boundaries": [asdict(item) for item in boundaries],
        "foreground_variations": [asdict(item) for item in variations],
        "harmony": [asdict(item) for item in harmony],
        "foreground_dissonance": [asdict(item) for item in dissonance],
        "rendered_harmony": asdict(rendered_harmony),
        "pedal_overlap_warning": asdict(pedal_warning),
        "pacing": {**asdict(pacing), "target_ratio": pacing_ratio},
        "ending": ending,
    }


def _load_reference_baseline() -> dict[str, object]:
    summary_path = REFERENCE_BASELINE_DIR / "summary.json"
    manifest_path = REFERENCE_BASELINE_DIR / "manifest.json"
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    checks = {
        "manifest_status_complete": manifest.get("status") == "complete",
        "axis_version_match": manifest.get("axis_version") == summary.get("axis_version"),
        "record_count_232": summary.get("record_count") == 232,
        "summary_hash_match": (sha256_file(summary_path) == manifest["outputs"]["summary.json"]),
    }
    if not all(checks.values()):
        raise ValueError(f"reference baseline validation failed: {checks}")
    axis = summary["axes"]["あかるさ"]
    return {
        "axis_version": summary["axis_version"],
        "record_count": summary["record_count"],
        "minimum": float(axis["minimum"]),
        "maximum": float(axis["maximum"]),
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(manifest_path),
        "checks": checks,
    }


def _normalized(raw: float, minimum: float, maximum: float) -> float:
    return 2 * (raw - minimum) / (maximum - minimum) - 1


def _score_structure_signature(score: ScoreSpec) -> dict[str, object]:
    return {
        material.material_id: {
            "foreground_voice": material.foreground_voice,
            "notes": [
                {
                    "event_id": note.event_id,
                    "at_units": note.at_units,
                    "duration_units": note.duration_units,
                    "voice": note.voice,
                    "tie": note.tie,
                    "articulations": list(note.articulations),
                }
                for note in material.notes
            ],
            "harmony_timing": [
                {
                    "at_units": harmony.at_units,
                    "duration_units": harmony.duration_units,
                }
                for harmony in material.harmonies
            ],
        }
        for material in score.materials
    }


def _section_brightness(rendered: RenderedPerformance, section: str) -> float:
    selected = tuple(
        note for note in rendered.notes if note.occurrence_node_id.startswith(f"{section}-")
    )
    piece = ReferencePiece.from_dicts(
        name=section,
        notes=(
            {
                "pitch": note.pitch,
                "onset_ms": note.at_ms,
                "duration_ms": note.duration_ms,
                "velocity": note.velocity,
            }
            for note in selected
        ),
        pedals=(),
    )
    return float(extract_control_observables(piece)["raw"]["あかるさ"])


def _machine_failed(output_dir: Path, **details) -> dict[str, object]:
    result = {
        "schema_version": 2,
        "status": "machine_failed",
        "scheme_id": SCHEME_ID,
        **details,
    }
    atomic_write_json(output_dir / "result.json", result)
    return _read_json(output_dir / "result.json")


def run_brightness_control(*, output_dir=DEFAULT_OUTPUT_DIR, input_dir=DEFAULT_INPUT_DIR):
    output_dir, input_dir = Path(output_dir), Path(input_dir)
    hashes = _input_hashes(input_dir)
    if hashes != EXPECTED_INPUT_HASHES:
        raise ValueError(f"input hash mismatch: expected {EXPECTED_INPUT_HASHES}, actual {hashes}")
    baseline = _load_reference_baseline()
    plan, score, performance = _load_inputs(input_dir)
    variants = build_brightness_variants(plan, score, performance)
    rendered = {
        level: render_performance(item.plan, item.score, item.performance)
        for level, item in variants.items()
    }
    base_by_id = {material.material_id: material for material in score.materials}
    quality = {}
    collisions = {}
    structure = {}
    for level in LEVELS:
        item = variants[level]
        structure[level] = _score_structure_signature(score) == _score_structure_signature(
            item.score
        )
        collisions[level] = {
            material.material_id: [
                list(pair)
                for pair in _new_or_worsened_voice_collisions(
                    base_by_id[material.material_id], material
                )
            ]
            for material in item.score.materials
        }
        collisions[level] = {key: value for key, value in collisions[level].items() if value}
        quality[level] = _quality_gate(
            item.plan,
            item.score,
            item.performance,
            rendered[level],
            score,
        )
        if not structure[level]:
            quality[level]["failures"].append("score-structure")
        if collisions[level]:
            quality[level]["failures"].append("new-or-worsened-voice-collisions")
        quality[level]["passes"] = not quality[level]["failures"]

    smf_bytes = {}
    smf_hashes = {}
    smf_metrics = {}
    common = {}
    with TemporaryDirectory(prefix="brightness-v2-") as temporary:
        temporary_dir = Path(temporary)
        for level in LEVELS:
            temporary_path = temporary_dir / f"{level}.mid"
            smf_metrics[level] = render_performance_smf(rendered[level], temporary_path)
            smf_bytes[level] = temporary_path.read_bytes()
            smf_hashes[level] = sha256_file(temporary_path)
            common[level] = extract_control_observables(load_reference_piece(temporary_path))

    if smf_hashes != EXPECTED_SMF_HASHES:
        return _machine_failed(
            output_dir,
            input_hashes=hashes,
            failure="diagnostic-endpoint-smf-mismatch",
            expected_smf_hashes=EXPECTED_SMF_HASHES,
            actual_smf_hashes=smf_hashes,
            quality_gates=quality,
        )
    if not all(item["passes"] for item in quality.values()):
        return _machine_failed(
            output_dir,
            input_hashes=hashes,
            failure="quality-gate",
            quality_gates=quality,
            voice_collisions=collisions,
            structure_preserved=structure,
        )

    declared = {
        level: generated_brightness_descriptor(variants[level].plan, rendered[level])
        for level in LEVELS
    }
    sections = {
        level: {
            section: _section_brightness(rendered[level], section) for section in ("a1", "b", "a2")
        }
        for level in LEVELS
    }
    direction_checks = {
        "common_coordinate": (common["low"]["raw"]["あかるさ"] < common["high"]["raw"]["あかるさ"]),
        "major_minor_profile_margin": (
            declared["low"]["metrics"]["major_minor_profile_margin"]["value"]
            < declared["high"]["metrics"]["major_minor_profile_margin"]["value"]
        ),
        "modal_degree_balance": (
            declared["low"]["metrics"]["modal_degree_balance"]["value"]
            < declared["high"]["metrics"]["modal_degree_balance"]["value"]
        ),
        "sections": all(
            sections["low"][section] < sections["high"][section] for section in ("a1", "b", "a2")
        ),
    }
    if not all(direction_checks.values()):
        return _machine_failed(
            output_dir,
            input_hashes=hashes,
            failure="brightness-direction",
            direction_checks=direction_checks,
            common_coordinate={level: common[level]["raw"] for level in LEVELS},
            declared_tonic=declared,
            sections=sections,
        )

    minimum = float(baseline["minimum"])
    maximum = float(baseline["maximum"])
    rows = []
    for level in LEVELS:
        path = output_dir / "variants" / f"{level}.mid"
        atomic_write_bytes(path, smf_bytes[level])
        raw = float(common[level]["raw"]["あかるさ"])
        requested = variants[level].requested
        target = minimum if requested == -1 else maximum
        covariates = {
            key: value for key, value in common[level]["raw"].items() if key != "あかるさ"
        }
        rows.append(
            {
                "level": level,
                "requested": requested,
                "resolution_status": "not_reached",
                "artifact_kind": "internal_calibration_endpoint",
                "tonal_plan_id": variants[level].tonal_plan_id,
                "target_raw": target,
                "achieved_raw": raw,
                "achieved_normalized": _normalized(raw, minimum, maximum),
                "raw_error": raw - target,
                "normalized_error": _normalized(raw, minimum, maximum) - requested,
                "smf": str(path.relative_to(output_dir)),
                "smf_sha256": sha256_file(path),
                "duration_ms": smf_metrics[level].duration_ms,
                "note_count": smf_metrics[level].note_count,
                "common_coordinate": {
                    "raw": common[level]["raw"],
                    "diagnostics": common[level]["diagnostics"],
                },
                "declared_tonic": declared[level],
                "sections": sections[level],
                "covariates": covariates,
                "repaired_event_ids": list(variants[level].repaired_event_ids),
                "transition_paths": [
                    asdict(path_item) for path_item in variants[level].transition_paths
                ],
                "structure_preserved": structure[level],
                "voice_collisions": collisions[level],
                "quality_gate": quality[level],
            }
        )
    result = {
        "schema_version": 2,
        "status": "machine_passed_listening_deferred",
        "scheme_id": SCHEME_ID,
        "middle_status": "not_implemented_pending_endpoint_listening",
        "input_hashes": hashes,
        "source_diagnostic_hashes": SOURCE_DIAGNOSTIC_HASHES,
        "reference_baseline": baseline,
        "direction_checks": direction_checks,
        "variants": rows,
    }
    atomic_write_json(output_dir / "result.json", result)
    return _read_json(output_dir / "result.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="あかるさ制御v2二端点を実行する")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    args = parser.parse_args(argv)
    result = run_brightness_control(output_dir=args.output_dir, input_dir=args.input_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str((args.output_dir / "result.json").resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 1 if result["status"] == "machine_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
