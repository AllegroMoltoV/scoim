"""採用済みv8から音域別再配置を含む高さ制御v2の機械対照を生成する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_musical_composer.brightness_control import height_descriptor
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
from llm_musical_composer.height_control import (
    SCHEME_ID,
    HeightVariant,
    build_height_candidate,
    candidate_transpositions,
    resolve_measured_height,
)
from llm_musical_composer.performance_pipeline import (
    RenderedPerformance,
    render_performance,
    render_performance_smf,
    validate_pipeline,
)
from llm_musical_composer.reference_profile import ReferencePiece
from llm_musical_composer.run_state import atomic_write_json, sha256_file
from llm_musical_composer.v8_quality import quality_gate_v8

ROOT = Path(__file__).parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / ".appendix" / "height-control-v2" / "runs"
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
        ROOT / "src" / "llm_musical_composer" / "brightness_control.py",
        ROOT / "src" / "llm_musical_composer" / "brightness_control_run.py",
        ROOT / "src" / "llm_musical_composer" / "control_reference_baseline.py",
        ROOT / "src" / "llm_musical_composer" / "height_control.py",
        ROOT / "src" / "llm_musical_composer" / "height_control_run.py",
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
        pedals=(
            {"at_ms": pedal.at_ms, "value": pedal.value} for pedal in rendered.pedals
        ),
    )


def _control_observables(
    name: str,
    rendered: RenderedPerformance,
    axes: dict[str, Any],
) -> dict[str, Any]:
    return normalize_record(
        extract_control_observables(_reference_piece(name, rendered)),
        axes,
        observation=True,
    )


def _row(level, variant, rendered, quality, controls) -> dict[str, Any]:
    return {
        "level": level,
        "purpose": variant.purpose,
        "resolution": asdict(variant.resolution),
        "revoicing": asdict(variant.diagnostics),
        "height": height_descriptor(rendered),
        "controls": {
            "raw": controls["raw"],
            "normalized": controls["normalized"],
            "range_status": controls["range_status"],
        },
        "quality_gate": quality,
    }


def run_height_control(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    input_dir: Path = DEFAULT_INPUT_DIR,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    baseline_manifest_path: Path = DEFAULT_BASELINE_MANIFEST_PATH,
) -> dict[str, Any]:
    """高さの到達可能候補と診断専用下端を生成する。"""

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
            "baseline hash mismatch: "
            f"expected {EXPECTED_BASELINE_HASHES}, actual {baseline_hashes}"
        )
    summary = _read_object(summary_path)
    if summary.get("axis_version") != "four-control-corpus-minmax-v3":
        raise ValueError("baseline axis version mismatch")
    axes = summary.get("axes")
    if not isinstance(axes, dict) or not isinstance(axes.get("高さ"), dict):
        raise ValueError("baseline summary is missing the height axis")
    height_axis = axes["高さ"]
    minimum = height_axis["minimum"]
    maximum = height_axis["maximum"]

    plan, score, performance = _load_inputs(input_dir)
    measured_candidates = {}
    rejected_candidates = {}
    for semitones in candidate_transpositions(score):
        try:
            candidate = build_height_candidate(
                plan, score, performance, semitones=semitones
            )
            validate_pipeline(candidate.plan, candidate.score, candidate.performance)
            candidate_rendered = render_performance(
                candidate.plan, candidate.score, candidate.performance
            )
            candidate_quality = quality_gate_v8(
                candidate.plan,
                candidate.score,
                candidate.performance,
                candidate_rendered,
                score,
            )
            if not candidate_quality["passes"]:
                rejected_candidates[str(semitones)] = candidate_quality["failures"]
                continue
            measured_candidates[semitones] = (
                candidate,
                candidate_rendered,
                candidate_quality,
                height_descriptor(candidate_rendered)["mean"],
            )
        except ValueError as error:
            rejected_candidates[str(semitones)] = [str(error)]
    resolutions = {
        level: resolve_measured_height(
            requested,
            candidate_means={
                semitones: item[3] for semitones, item in measured_candidates.items()
            },
            minimum=minimum,
            maximum=maximum,
        )
        for level, requested in REQUESTS.items()
    }
    variants = {}
    rendered = {}
    quality = {}
    for level, resolution in resolutions.items():
        candidate, candidate_rendered, candidate_quality, _ = measured_candidates[
            resolution.semitones
        ]
        variants[level] = HeightVariant(
            scheme_id=SCHEME_ID,
            purpose=(
                "diagnostic_only"
                if resolution.status == "unreachable"
                else "candidate"
            ),
            resolution=resolution,
            plan=candidate.plan,
            score=candidate.score,
            performance=candidate.performance,
            diagnostics=candidate.diagnostics,
        )
        rendered[level] = candidate_rendered
        quality[level] = candidate_quality
    controls = {
        level: _control_observables(f"height-{level}.mid", rendered[level], axes)
        for level in REQUESTS
    }
    descriptors = {level: height_descriptor(rendered[level]) for level in REQUESTS}
    for metric in ("mean", "median", "p10", "p90"):
        values = [descriptors[level][metric] for level in REQUESTS]
        if values != sorted(values) or len(set(values)) != len(values):
            quality["high"]["failures"].append(f"height-order:{metric}")
            quality["high"]["passes"] = False
    voices = set(descriptors["low"]["voice_means"])
    for voice in voices:
        values = [descriptors[level]["voice_means"][voice] for level in REQUESTS]
        if values != sorted(values) or len(set(values)) != len(values):
            quality["high"]["failures"].append(f"height-order:voice:{voice}")
            quality["high"]["passes"] = False
    covariation = {
        axis: {
            level: controls[level]["raw"][axis] for level in REQUESTS
        }
        for axis in ("あかるさ", "重なり", "発音頻度")
    }
    if len(set(covariation["発音頻度"].values())) != 1:
        quality["high"]["failures"].append("unexpected-covariation:発音頻度")
        quality["high"]["passes"] = False

    implementation_hashes = _implementation_hashes()
    run_id = _run_id(input_hashes, baseline_hashes, implementation_hashes)
    run_dir = output_root / run_id
    rows = {
        level: _row(level, variants[level], rendered[level], quality[level], controls[level])
        for level in REQUESTS
    }
    if not all(item["passes"] for item in quality.values()):
        result = {
            "schema_version": 2,
            "scheme_id": SCHEME_ID,
            "status": "machine_failed",
            "run_id": run_id,
            "input_hashes": input_hashes,
            "baseline_hashes": baseline_hashes,
            "quality_gates": quality,
            "rejected_candidates": rejected_candidates,
        }
        atomic_write_json(run_dir / "result.json", result)
        return {**result, "run_dir": str(run_dir.resolve())}

    smf_paths = {
        "low": run_dir / "diagnostics" / "low-reachable-boundary.mid",
        "middle": run_dir / "variants" / "middle.mid",
        "high": run_dir / "variants" / "high.mid",
    }
    for level, path in smf_paths.items():
        smf = render_performance_smf(rendered[level], path)
        rows[level].update(
            {
                "smf": path.relative_to(run_dir).as_posix(),
                "smf_sha256": sha256_file(path),
                "duration_ms": smf.duration_ms,
                "note_count": smf.note_count,
            }
        )
    result = {
        "schema_version": 2,
        "scheme_id": SCHEME_ID,
        "status": "machine_passed_with_unreachable",
        "run_id": run_id,
        "input_hashes": input_hashes,
        "baseline_hashes": baseline_hashes,
        "candidate_count": len(measured_candidates),
        "rejected_candidates": rejected_candidates,
        "covariation": covariation,
        "diagnostics": [rows["low"]],
        "variants": [rows["middle"], rows["high"]],
    }
    atomic_write_json(run_dir / "result.json", result)
    output_hashes = {
        rows[level]["smf"]: rows[level]["smf_sha256"] for level in REQUESTS
    }
    output_hashes["result.json"] = sha256_file(run_dir / "result.json")
    manifest = {
        "schema_version": 2,
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
    parser.add_argument(
        "--baseline-manifest", type=Path, default=DEFAULT_BASELINE_MANIFEST_PATH
    )
    args = parser.parse_args(argv)
    result = run_height_control(
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
