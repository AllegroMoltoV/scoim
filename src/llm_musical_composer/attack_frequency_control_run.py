"""採用済みv8から30 ms発音群による発音頻度制御v1を生成する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from llm_musical_composer.attack_frequency_control import (
    PROTECTED_MATERIALS,
    SCHEME_ID,
    FrequencyScoreCandidate,
    MeasuredFrequencyCandidate,
    build_high_frequency_candidate,
    build_low_frequency_candidate,
    resolve_measured_frequency,
    scale_score_resolution,
)
from llm_musical_composer.brightness_control_run import (
    DEFAULT_INPUT_DIR,
    EXPECTED_INPUT_HASHES,
    _input_hashes,
    _load_inputs,
)
from llm_musical_composer.control_reference_baseline import (
    extract_control_observables,
    normalize_record,
)
from llm_musical_composer.performance_pipeline import (
    RenderedPerformance,
    ScoreSpec,
    render_performance,
    render_performance_smf,
    validate_pipeline,
)
from llm_musical_composer.reference_profile import ReferencePiece
from llm_musical_composer.run_state import atomic_write_json, sha256_file
from llm_musical_composer.v8_quality import quality_gate_v8

ROOT = Path(__file__).parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / ".appendix" / "attack-frequency-control-v1" / "runs"
DEFAULT_SUMMARY_PATH = ROOT / ".appendix" / "control-reference-baseline-v3" / "summary.json"
DEFAULT_BASELINE_MANIFEST_PATH = (
    ROOT / ".appendix" / "control-reference-baseline-v3" / "manifest.json"
)
EXPECTED_BASELINE_HASHES = {
    "manifest": "18726739adee11077f4246aea4f1e3836413f3aa2a90224705ae56590227b9ea",
    "summary": "dbbda38ea4ee25a1c3371a8487968b3f417454d0e6513046d5b71000259c56a7",
}
REQUESTS = {"low": -1.0, "middle": 0.0, "high": 1.0}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def _baseline_hashes(summary_path: Path, manifest_path: Path) -> dict[str, str]:
    return {
        "manifest": sha256_file(manifest_path),
        "summary": sha256_file(summary_path),
    }


def _implementation_hashes() -> dict[str, str]:
    paths = (
        ROOT / "src" / "llm_musical_composer" / "attack_frequency_control.py",
        ROOT / "src" / "llm_musical_composer" / "attack_frequency_control_run.py",
        ROOT / "src" / "llm_musical_composer" / "brightness_control_run.py",
        ROOT / "src" / "llm_musical_composer" / "control_reference_baseline.py",
        ROOT / "src" / "llm_musical_composer" / "performance_pipeline.py",
        ROOT / "src" / "llm_musical_composer" / "pipeline_dsl.py",
        ROOT / "src" / "llm_musical_composer" / "recurrence_quality.py",
        ROOT / "src" / "llm_musical_composer" / "reference_profile.py",
        ROOT / "src" / "llm_musical_composer" / "v8_quality.py",
    )
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def _run_id(
    input_hashes: dict[str, str],
    baseline_hashes: dict[str, str],
    implementation_hashes: dict[str, str],
) -> str:
    payload = json.dumps(
        {
            "scheme_id": SCHEME_ID,
            "input_hashes": input_hashes,
            "baseline_hashes": baseline_hashes,
            "implementation_hashes": implementation_hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _reference_piece(name: str, rendered: RenderedPerformance) -> ReferencePiece:
    return ReferencePiece.from_dicts(
        name=name,
        notes=(
            {
                "pitch": note.pitch,
                "onset_ms": note.at_ms,
                "duration_ms": note.duration_ms,
                "velocity": note.velocity,
            }
            for note in rendered.notes
        ),
        pedals=({"at_ms": pedal.at_ms, "value": pedal.value} for pedal in rendered.pedals),
    )


def _controls(name: str, rendered: RenderedPerformance, axes: dict[str, Any]) -> dict[str, Any]:
    return normalize_record(
        extract_control_observables(_reference_piece(name, rendered)),
        axes,
        observation=True,
    )


def _resolution_drift(
    base: RenderedPerformance,
    scaled: RenderedPerformance,
    base_controls: dict[str, Any],
    scaled_controls: dict[str, Any],
) -> dict[str, Any]:
    base_notes = {(note.occurrence_node_id, note.event_id): note for note in base.notes}
    scaled_notes = {(note.occurrence_node_id, note.event_id): note for note in scaled.notes}
    if base_notes.keys() != scaled_notes.keys():
        raise ValueError("resolution scaling changed note ids")
    note_time_delta = max(
        max(
            abs(base_notes[event_id].at_ms - scaled_notes[event_id].at_ms),
            abs(base_notes[event_id].duration_ms - scaled_notes[event_id].duration_ms),
        )
        for event_id in base_notes
    )
    if any(
        replace(
            scaled_notes[event_id],
            at_ms=base_notes[event_id].at_ms,
            duration_ms=base_notes[event_id].duration_ms,
        )
        != base_notes[event_id]
        for event_id in base_notes
    ):
        raise ValueError("resolution scaling changed note identity")
    base_pedals = {(pedal.occurrence_node_id, pedal.event_id): pedal for pedal in base.pedals}
    scaled_pedals = {(pedal.occurrence_node_id, pedal.event_id): pedal for pedal in scaled.pedals}
    if base_pedals.keys() != scaled_pedals.keys():
        raise ValueError("resolution scaling changed pedal ids")
    pedal_time_delta = max(
        abs(base_pedals[event_id].at_ms - scaled_pedals[event_id].at_ms) for event_id in base_pedals
    )
    if any(
        replace(scaled_pedals[event_id], at_ms=base_pedals[event_id].at_ms) != base_pedals[event_id]
        for event_id in base_pedals
    ):
        raise ValueError("resolution scaling changed pedal identity")
    raw_deltas = {
        axis: abs(scaled_controls["raw"][axis] - base_controls["raw"][axis])
        for axis in ("あかるさ", "高さ", "重なり", "発音頻度")
    }
    return {
        "maximum_note_time_delta_ms": note_time_delta,
        "maximum_pedal_time_delta_ms": pedal_time_delta,
        "raw_control_deltas": raw_deltas,
        "passes": bool(
            note_time_delta <= 1
            and pedal_time_delta <= 1
            and raw_deltas["高さ"] == 0
            and raw_deltas["発音頻度"] == 0
            and raw_deltas["あかるさ"] < 2e-5
            and raw_deltas["重なり"] < 2e-5
        ),
    }


def _score_invariants(
    base: ScoreSpec,
    candidate: ScoreSpec,
    transform: FrequencyScoreCandidate | None,
) -> list[str]:
    failures: list[str] = []
    originals = {material.material_id: material for material in base.materials}
    for material in candidate.materials:
        original = originals[material.material_id]
        if material.material_id in PROTECTED_MATERIALS and material != original:
            failures.append(f"protected:{material.material_id}")
        if replace(material, notes=original.notes) != original:
            failures.append(f"material-metadata:{material.material_id}")
        original_notes = {note.event_id: note for note in original.notes}
        candidate_notes = {note.event_id: note for note in material.notes}
        for event_id, note in original_notes.items():
            if event_id not in candidate_notes:
                failures.append(f"missing:{event_id}")
                continue
            changed = candidate_notes[event_id]
            if transform is not None and transform.direction == "low":
                changed = replace(changed, at_units=note.at_units)
            if changed != note:
                failures.append(f"note:{event_id}")
        extras = set(candidate_notes) - set(original_notes)
        if transform is None or transform.direction != "high":
            if extras:
                failures.append(f"extra:{material.material_id}")
        elif any(not event_id.startswith("freq+") for event_id in extras):
            failures.append(f"extra-id:{material.material_id}")
    return failures


def _quality(
    plan,
    score,
    performance,
    rendered,
    base_score,
    *,
    allow_note_count: bool,
) -> dict[str, Any]:
    quality = quality_gate_v8(plan, score, performance, rendered, base_score)
    raw_failures = list(quality["failures"])
    excluded = [failure for failure in raw_failures if allow_note_count and failure == "note-count"]
    failures = [failure for failure in raw_failures if failure not in excluded]
    return {
        "passes": not failures,
        "failures": failures,
        "raw_failures": raw_failures,
        "excluded_failures": excluded,
    }


def run_attack_frequency_control(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    input_dir: Path = DEFAULT_INPUT_DIR,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    baseline_manifest_path: Path = DEFAULT_BASELINE_MANIFEST_PATH,
) -> dict[str, Any]:
    """6候補を実測し、低・中・高の発音頻度対照を生成する。"""

    output_root = Path(output_root)
    input_dir = Path(input_dir)
    summary_path = Path(summary_path)
    baseline_manifest_path = Path(baseline_manifest_path)
    input_hashes = _input_hashes(input_dir)
    if input_hashes != EXPECTED_INPUT_HASHES:
        raise ValueError(
            f"input hash mismatch: expected {EXPECTED_INPUT_HASHES}, actual {input_hashes}"
        )
    baseline_hashes = _baseline_hashes(summary_path, baseline_manifest_path)
    if baseline_hashes != EXPECTED_BASELINE_HASHES:
        raise ValueError(
            f"baseline hash mismatch: expected {EXPECTED_BASELINE_HASHES}, actual {baseline_hashes}"
        )
    summary = _read_object(summary_path)
    if summary.get("axis_version") != "four-control-corpus-minmax-v3":
        raise ValueError("baseline axis version mismatch")
    axes = summary.get("axes")
    if not isinstance(axes, dict) or not isinstance(axes.get("発音頻度"), dict):
        raise ValueError("baseline summary is missing the attack frequency axis")
    minimum = axes["発音頻度"]["minimum"]
    maximum = axes["発音頻度"]["maximum"]

    plan, source_score, performance = _load_inputs(input_dir)
    source_rendered = render_performance(plan, source_score, performance)
    source_controls = _controls("source.mid", source_rendered, axes)
    base_score = scale_score_resolution(source_score)
    validate_pipeline(plan, base_score, performance)
    base_rendered = render_performance(plan, base_score, performance)
    base_controls = _controls("base.mid", base_rendered, axes)
    drift = _resolution_drift(source_rendered, base_rendered, source_controls, base_controls)
    if not drift["passes"]:
        raise ValueError(f"resolution scaling drift exceeded contract: {drift}")

    catalog: list[tuple[str, FrequencyScoreCandidate | None]] = [("base", None)]
    catalog.extend(
        (f"low-{stage}", build_low_frequency_candidate(base_score, stage=stage))
        for stage in range(1, 3)
    )
    catalog.extend(
        (f"high-{stage}", build_high_frequency_candidate(base_score, stage=stage))
        for stage in range(1, 4)
    )
    candidates: dict[str, dict[str, Any]] = {}
    measured: list[MeasuredFrequencyCandidate] = []
    for candidate_id, transform in catalog:
        score = transform.score if transform is not None else base_score
        validate_pipeline(plan, score, performance)
        rendered = render_performance(plan, score, performance)
        controls = _controls(f"{candidate_id}.mid", rendered, axes)
        quality = _quality(
            plan,
            score,
            performance,
            rendered,
            base_score,
            allow_note_count=bool(transform and transform.direction == "high"),
        )
        invariant_failures = _score_invariants(base_score, score, transform)
        if rendered.pedals != base_rendered.pedals:
            invariant_failures.append("pedals")
        frequency_movement = abs(
            controls["normalized"]["発音頻度"] - base_controls["normalized"]["発音頻度"]
        )
        other_movements = {
            axis: abs(controls["normalized"][axis] - base_controls["normalized"][axis])
            for axis in ("あかるさ", "高さ", "重なり")
        }
        if any(value > frequency_movement + 1e-12 for value in other_movements.values()):
            invariant_failures.append("cross-axis-dominance")
        safe = bool(quality["passes"] and not invariant_failures)
        changed_note_count = transform.changed_note_count if transform else 0
        frequency = controls["raw"]["発音頻度"]
        measured.append(
            MeasuredFrequencyCandidate(candidate_id, frequency, safe, changed_note_count)
        )
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "safe": safe,
            "changed_note_count": changed_note_count,
            "note_count": len(rendered.notes),
            "attack_group_count": controls["diagnostics"]["attack_group_count"],
            "quality_gate": quality,
            "invariant_failures": invariant_failures,
            "cross_axis_movement": other_movements,
            "controls": controls,
            "score": score,
            "rendered": rendered,
        }
    base_frequency = candidates["base"]["controls"]["raw"]["発音頻度"]
    low_frequencies = [
        candidates[candidate_id]["controls"]["raw"]["発音頻度"]
        for candidate_id in ("low-2", "low-1")
    ] + [base_frequency]
    high_frequencies = [base_frequency] + [
        candidates[candidate_id]["controls"]["raw"]["発音頻度"]
        for candidate_id in ("high-1", "high-2", "high-3")
    ]
    if low_frequencies != sorted(low_frequencies) or high_frequencies != sorted(high_frequencies):
        raise ValueError("frequency candidates are not monotonic")

    resolutions = {
        level: resolve_measured_frequency(
            requested,
            candidates=tuple(measured),
            minimum=minimum,
            maximum=maximum,
        )
        for level, requested in REQUESTS.items()
    }
    selected = {
        level: candidates[resolution.candidate_id] for level, resolution in resolutions.items()
    }
    output_rows = []
    for level in REQUESTS:
        candidate = selected[level]
        output_rows.append(
            {
                "level": level,
                "resolution": asdict(resolutions[level]),
                "candidate_id": candidate["candidate_id"],
                "safe": candidate["safe"],
                "changed_note_count": candidate["changed_note_count"],
                "controls": {
                    "raw": candidate["controls"]["raw"],
                    "normalized": candidate["controls"]["normalized"],
                    "range_status": candidate["controls"]["range_status"],
                },
                "quality_gate": candidate["quality_gate"],
                "invariant_failures": candidate["invariant_failures"],
            }
        )

    implementation_hashes = _implementation_hashes()
    run_id = _run_id(input_hashes, baseline_hashes, implementation_hashes)
    run_dir = output_root / run_id
    status = (
        "machine_failed"
        if not all(row["safe"] for row in output_rows)
        else "machine_passed_with_unreachable"
        if any(row["resolution"]["status"] == "unreachable" for row in output_rows)
        else "machine_passed"
    )
    if status != "machine_failed":
        for row in output_rows:
            path = run_dir / "outputs" / f"{row['level']}.mid"
            smf = render_performance_smf(selected[row["level"]]["rendered"], path)
            row.update(
                {
                    "smf": path.relative_to(run_dir).as_posix(),
                    "smf_sha256": sha256_file(path),
                    "duration_ms": smf.duration_ms,
                    "note_count": smf.note_count,
                }
            )
    candidate_rows = [
        {
            key: value
            for key, value in candidates[candidate_id].items()
            if key not in {"controls", "score", "rendered"}
        }
        | {
            "controls": {
                "raw": candidates[candidate_id]["controls"]["raw"],
                "normalized": candidates[candidate_id]["controls"]["normalized"],
            }
        }
        for candidate_id, _ in catalog
    ]
    result = {
        "schema_version": 1,
        "scheme_id": SCHEME_ID,
        "status": status,
        "run_id": run_id,
        "input_hashes": input_hashes,
        "baseline_hashes": baseline_hashes,
        "reference_frequency_range": {"minimum": minimum, "maximum": maximum},
        "resolution_drift": drift,
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "outputs": output_rows,
    }
    atomic_write_json(run_dir / "result.json", result)
    if status != "machine_failed":
        output_hashes = {row["smf"]: row["smf_sha256"] for row in output_rows}
        output_hashes["result.json"] = sha256_file(run_dir / "result.json")
        manifest = {
            "schema_version": 1,
            "scheme_id": SCHEME_ID,
            "status": "complete",
            "run_id": run_id,
            "input_hashes": input_hashes,
            "baseline_hashes": baseline_hashes,
            "implementation_hashes": implementation_hashes,
            "outputs": output_hashes,
        }
        atomic_write_json(run_dir / "manifest.json", manifest)
    return {**result, "run_dir": str(run_dir.resolve())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE_MANIFEST_PATH)
    args = parser.parse_args(argv)
    result = run_attack_frequency_control(
        output_root=args.output_root,
        input_dir=args.input_dir,
        summary_path=args.summary,
        baseline_manifest_path=args.baseline_manifest,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_dir": result["run_dir"],
                "result": str((Path(result["run_dir"]) / "result.json").resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if result["status"] == "machine_failed" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
