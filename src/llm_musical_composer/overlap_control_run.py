"""採用済みv8から役割別音価による重なり制御v1を生成する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

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
from llm_musical_composer.overlap_control import (
    CANDIDATE_SPECS,
    PARENT_NODES,
    PROTECTED_MATERIALS,
    SCHEME_ID,
    MeasuredOverlapCandidate,
    build_performance_candidate,
    build_score_candidate,
    resolve_measured_overlap,
)
from llm_musical_composer.performance_pipeline import (
    PerformanceSpec,
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
DEFAULT_OUTPUT_ROOT = ROOT / ".appendix" / "overlap-control-v1" / "runs"
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
        ROOT / "src" / "llm_musical_composer" / "brightness_control_run.py",
        ROOT / "src" / "llm_musical_composer" / "control_reference_baseline.py",
        ROOT / "src" / "llm_musical_composer" / "overlap_control.py",
        ROOT / "src" / "llm_musical_composer" / "overlap_control_run.py",
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


def _score_invariants(base: ScoreSpec, candidate: ScoreSpec) -> list[str]:
    failures: list[str] = []
    if candidate.divisions != base.divisions:
        failures.append("divisions")
    originals = {material.material_id: material for material in base.materials}
    if {item.material_id for item in candidate.materials} != set(originals):
        failures.append("material-set")
        return failures
    for material in candidate.materials:
        original = originals[material.material_id]
        if material.material_id in PROTECTED_MATERIALS and material != original:
            failures.append(f"protected:{material.material_id}")
        if replace(material, notes=original.notes) != original:
            failures.append(f"material-metadata:{material.material_id}")
        if len(material.notes) != len(original.notes):
            failures.append(f"note-count:{material.material_id}")
            continue
        for before, after in zip(original.notes, material.notes, strict=True):
            if replace(after, duration_units=before.duration_units) != before:
                failures.append(f"note-metadata:{material.material_id}:{before.event_id}")
    return failures


def _performance_invariants(
    base: PerformanceSpec, candidate: PerformanceSpec, *, profile: str | None
) -> list[str]:
    if profile is None:
        return [] if candidate == base else ["performance"]
    failures: list[str] = []
    if replace(candidate, performance_id=base.performance_id) == base:
        return failures
    before = {item.node_id: item for item in base.node_performances}
    after = {item.node_id: item for item in candidate.node_performances}
    if before.keys() != after.keys():
        failures.append("node-set")
        return failures
    for node_id, original in before.items():
        changed = after[node_id]
        if node_id in PARENT_NODES:
            if replace(changed, articulation_profile=original.articulation_profile) != original:
                failures.append(f"parent:{node_id}")
        elif changed != original:
            failures.append(f"child:{node_id}")
    if (
        candidate.target_duration_ms != base.target_duration_ms
        or candidate.default_velocity != base.default_velocity
        or candidate.timing_budget_id != base.timing_budget_id
    ):
        failures.append("performance-metadata")
    return failures


def _rendered_invariants(base: RenderedPerformance, candidate: RenderedPerformance) -> list[str]:
    failures: list[str] = []
    if base.pedals != candidate.pedals:
        failures.append("pedals")
    if len(base.notes) != len(candidate.notes):
        failures.append("note-count")
        return failures
    for before, after in zip(base.notes, candidate.notes, strict=True):
        if replace(after, duration_ms=before.duration_ms) != before:
            failures.append(f"note:{before.event_id}")
    return failures


def _harmony_crossings(score: ScoreSpec) -> list[str]:
    failures: list[str] = []
    for material in score.materials:
        for note in material.notes:
            matches = tuple(
                harmony
                for harmony in material.harmonies
                if harmony.at_units <= note.at_units < harmony.at_units + harmony.duration_units
            )
            if (
                len(matches) == 1
                and note.at_units + note.duration_units
                > matches[0].at_units + matches[0].duration_units
            ):
                failures.append(f"{material.material_id}:{note.event_id}")
    return failures


def _profile_change_count(base: PerformanceSpec, candidate: PerformanceSpec) -> int:
    before = {item.node_id: item for item in base.node_performances}
    return sum(
        item.node_id in PARENT_NODES
        and item.articulation_profile != before[item.node_id].articulation_profile
        for item in candidate.node_performances
    )


def run_overlap_control(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    input_dir: Path = DEFAULT_INPUT_DIR,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    baseline_manifest_path: Path = DEFAULT_BASELINE_MANIFEST_PATH,
) -> dict[str, Any]:
    """18候補を実測し、低・中・高の重なり対照を生成する。"""

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
    if not isinstance(axes, dict) or not isinstance(axes.get("重なり"), dict):
        raise ValueError("baseline summary is missing the overlap axis")
    minimum = axes["重なり"]["minimum"]
    maximum = axes["重なり"]["maximum"]

    plan, base_score, base_performance = _load_inputs(input_dir)
    base_rendered = render_performance(plan, base_score, base_performance)
    base_controls = _controls("overlap-base.mid", base_rendered, axes)
    candidates: dict[str, dict[str, Any]] = {}
    measurements: list[MeasuredOverlapCandidate] = []
    for spec in CANDIDATE_SPECS:
        score_candidate = (
            build_score_candidate(base_score, direction=spec.direction, stage=spec.stage)
            if spec.direction is not None and spec.stage is not None
            else None
        )
        candidate_score = score_candidate.score if score_candidate is not None else base_score
        candidate_performance = (
            build_performance_candidate(base_performance, profile=spec.articulation_profile)
            if spec.articulation_profile is not None
            else base_performance
        )
        validate_pipeline(plan, candidate_score, candidate_performance)
        rendered = render_performance(plan, candidate_score, candidate_performance)
        controls = _controls(f"{spec.candidate_id}.mid", rendered, axes)
        quality = quality_gate_v8(
            plan,
            candidate_score,
            candidate_performance,
            rendered,
            base_score,
        )
        invariant_failures = (
            _score_invariants(base_score, candidate_score)
            + _performance_invariants(
                base_performance,
                candidate_performance,
                profile=spec.articulation_profile,
            )
            + _rendered_invariants(base_rendered, rendered)
        )
        crossings = _harmony_crossings(candidate_score)
        if crossings:
            invariant_failures.extend(f"harmony-crossing:{item}" for item in crossings)
        if (
            controls["diagnostics"]["attack_group_count"]
            != base_controls["diagnostics"]["attack_group_count"]
        ):
            invariant_failures.append("attack-group-count")
        changed_note_count = (
            score_candidate.changed_note_count if score_candidate is not None else 0
        )
        profile_change_count = _profile_change_count(base_performance, candidate_performance)
        raw_overlap = controls["raw"]["重なり"]
        safe = bool(quality["passes"] and not invariant_failures)
        measurements.append(
            MeasuredOverlapCandidate(
                spec.candidate_id,
                raw_overlap,
                safe,
                changed_note_count,
                profile_change_count,
            )
        )
        candidates[spec.candidate_id] = {
            "candidate_id": spec.candidate_id,
            "raw_overlap": raw_overlap,
            "normalized_overlap": controls["normalized"]["重なり"],
            "safe": safe,
            "changed_note_count": changed_note_count,
            "profile_change_count": profile_change_count,
            "changed_role_counts": dict(
                score_candidate.changed_role_counts if score_candidate else ()
            ),
            "quality_gate": quality,
            "invariant_failures": invariant_failures,
            "score": candidate_score,
            "performance": candidate_performance,
            "rendered": rendered,
            "controls": controls,
        }

    resolutions = {
        level: resolve_measured_overlap(
            requested,
            candidates=tuple(measurements),
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
                "profile_change_count": candidate["profile_change_count"],
                "changed_role_counts": candidate["changed_role_counts"],
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
            for key, value in candidates[spec.candidate_id].items()
            if key not in {"score", "performance", "rendered", "controls"}
        }
        for spec in CANDIDATE_SPECS
    ]
    result = {
        "schema_version": 1,
        "scheme_id": SCHEME_ID,
        "status": status,
        "run_id": run_id,
        "input_hashes": input_hashes,
        "baseline_hashes": baseline_hashes,
        "reference_overlap_range": {"minimum": minimum, "maximum": maximum},
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
    result = run_overlap_control(
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
