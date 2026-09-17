"""既存の3段IRへ内部の高さ軸を適用し、品質確認済みSMFを生成する。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_musical_composer.brightness_control import height_descriptor
from llm_musical_composer.composition_request import (
    normalize_composition_request,
    resolve_composition_request,
)
from llm_musical_composer.control_reference_baseline import (
    extract_control_observables,
    normalize_record,
)
from llm_musical_composer.generic_pipeline_quality import (
    evaluate_generic_pipeline_quality,
)
from llm_musical_composer.height_control import (
    SCHEME_ID,
    HeightResolution,
    build_height_candidate,
    candidate_transpositions,
    resolve_measured_height,
    validate_height_candidate_contract,
)
from llm_musical_composer.performance_pipeline import (
    RenderedPerformance,
    render_performance,
    render_performance_smf,
    validate_pipeline,
    validate_smf_round_trip,
)
from llm_musical_composer.pipeline_dsl import (
    dump_performance_spec,
    dump_piece_plan,
    dump_score_spec,
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.reference_generation_target import (
    build_reference_generation_target,
)
from llm_musical_composer.reference_profile import ReferencePiece
from llm_musical_composer.run_state import (
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    sha256_json,
)

ROOT = Path(__file__).parents[2]
IMPLEMENTATION_PATHS = (
    "src/llm_musical_composer/brightness_control.py",
    "src/llm_musical_composer/composition_request.py",
    "src/llm_musical_composer/control_reference_baseline.py",
    "src/llm_musical_composer/generic_pipeline_quality.py",
    "src/llm_musical_composer/height_control.py",
    "src/llm_musical_composer/height_control_integration_run.py",
    "src/llm_musical_composer/performance_pipeline.py",
    "src/llm_musical_composer/pipeline_dsl.py",
    "src/llm_musical_composer/reference_generation_target.py",
    "src/llm_musical_composer/reference_profile.py",
    "src/llm_musical_composer/run_state.py",
)


class HeightIntegrationError(RuntimeError):
    """高さ軸の通し実行を安全に確定できない場合に送出する。"""


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HeightIntegrationError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise HeightIntegrationError(f"{label} must be a JSON object")
    return value


def _repo_path(relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise HeightIntegrationError(f"{label} path is invalid")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise HeightIntegrationError(f"{label} path is outside repository") from error
    return path


def _baseline_path(baseline_root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise HeightIntegrationError(f"{label} path is invalid")
    path = (baseline_root / relative).resolve()
    try:
        path.relative_to(baseline_root.resolve())
    except ValueError as error:
        raise HeightIntegrationError(f"{label} path is outside baseline") from error
    return path


def _declared_file(record: object, label: str) -> tuple[Path, str]:
    if not isinstance(record, Mapping):
        raise HeightIntegrationError(f"{label} declaration is invalid")
    path = _repo_path(record.get("path"), label)
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise HeightIntegrationError(f"{label} hash is invalid")
    if not path.is_file():
        raise HeightIntegrationError(f"{label} is missing")
    actual = sha256_file(path)
    if actual.casefold() != digest.casefold():
        raise HeightIntegrationError(f"{label} hash mismatch")
    return path, actual


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise HeightIntegrationError(f"missing provenance field: {'.'.join(keys)}")
        value = value[key]
    return value


def _validated_inputs(input_spec_path: Path) -> dict[str, Any]:
    spec = _read_object(input_spec_path, "input spec")
    if spec.get("schema_version") != 1:
        raise HeightIntegrationError("input spec schema_version must be 1")
    run_spec_path, run_spec_hash = _declared_file(spec.get("run_spec"), "run spec")
    manifest_path, manifest_hash = _declared_file(spec.get("manifest"), "manifest")
    run_spec = _read_object(run_spec_path, "run spec")
    manifest = _read_object(manifest_path, "manifest")

    baseline_root = _repo_path(spec.get("baseline_root"), "baseline root")
    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise HeightIntegrationError("input spec artifacts are invalid")
    paths = {
        key: _baseline_path(baseline_root, artifacts.get(key), key)
        for key in ("piece_plan", "score_spec", "performance_spec", "final_smf")
    }
    expected = {
        "piece_plan": _nested(
            run_spec, "input_hashes", "generation_inputs", "piece_plan", "sha256"
        ),
        "score_spec": _nested(manifest, "source", "score_spec_sha256"),
        "performance_spec": _nested(
            manifest, "outputs", "performance_spec_sha256"
        ),
        "final_smf": _nested(manifest, "outputs", "smf_sha256"),
    }
    hashes: dict[str, str] = {}
    for key, path in paths.items():
        if not path.is_file():
            raise HeightIntegrationError(f"baseline artifact is missing: {key}")
        actual = sha256_file(path)
        if not isinstance(expected[key], str) or actual.casefold() != expected[key].casefold():
            raise HeightIntegrationError(f"baseline artifact hash mismatch: {key}")
        hashes[key] = actual

    resolved_declared = _nested(
        run_spec, "input_hashes", "generation_inputs", "resolved_request"
    )
    resolved_path, resolved_hash = _declared_file(
        resolved_declared, "baseline resolved request"
    )
    resolved_request = _read_object(resolved_path, "baseline resolved request")
    reference = resolved_request.get("reference")
    if not isinstance(reference, Mapping) or reference.get("state") != "resolved":
        raise HeightIntegrationError("baseline reference provenance is unresolved")

    return {
        "spec": spec,
        "run_spec_hash": run_spec_hash,
        "manifest_hash": manifest_hash,
        "artifact_paths": paths,
        "artifact_hashes": hashes,
        "baseline_resolved_request": resolved_request,
        "baseline_resolved_request_hash": resolved_hash,
    }


def _implementation_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in IMPLEMENTATION_PATHS}


def build_run_id(
    semantic_inputs: Mapping[str, Any], implementation_hashes: Mapping[str, str]
) -> str:
    """出力先に依存しない意味入力と実装からrun IDを作る。"""

    return sha256_json(
        {
            "scheme_id": SCHEME_ID,
            "semantic_inputs": dict(semantic_inputs),
            "implementation_hashes": dict(sorted(implementation_hashes.items())),
        }
    )[:16]


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
            {"at_ms": pedal.at_ms, "value": pedal.value}
            for pedal in rendered.pedals
        ),
    )


def _controls(
    name: str, rendered: RenderedPerformance, axes: Mapping[str, Any]
) -> dict[str, Any]:
    record = normalize_record(
        extract_control_observables(_reference_piece(name, rendered)),
        axes,
        observation=True,
    )
    return {
        "raw": record["raw"],
        "normalized": record["normalized"],
        "range_status": record["range_status"],
    }


def _portable_round_trip(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("path", None)
    result["path_role"] = "generated_smf"
    return result


def resolve_accepted_height(
    requested: float,
    *,
    observations: list[Mapping[str, Any]],
    minimum: float,
    maximum: float,
) -> HeightResolution:
    """全検査を通過した候補だけで高さ要求を解決する。"""

    candidates = {
        int(item["semitones"]): float(item["height_mean"])
        for item in observations
        if item.get("accepted") is True
    }
    return resolve_measured_height(
        requested,
        candidate_means=candidates,
        minimum=minimum,
        maximum=maximum,
    )


def _verify_complete_run(run_dir: Path) -> dict[str, Any] | None:
    if not run_dir.exists():
        return None
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HeightIntegrationError(f"incomplete run already exists: {run_dir.name}")
    manifest = _read_object(manifest_path, "output manifest")
    outputs = manifest.get("outputs")
    if manifest.get("status") != "complete" or not isinstance(outputs, Mapping):
        raise HeightIntegrationError(f"invalid completed run: {run_dir.name}")
    for relative, expected in outputs.items():
        path = _baseline_path(run_dir, relative, "output")
        if not path.is_file() or not isinstance(expected, str):
            raise HeightIntegrationError(f"completed run output is missing: {relative}")
        if sha256_file(path).casefold() != expected.casefold():
            raise HeightIntegrationError(f"completed run output hash mismatch: {relative}")
    result = _read_object(run_dir / "result.json", "saved result")
    return {**result, "run_dir": str(run_dir.resolve())}


def run_height_integration(
    *,
    request_path: Path,
    input_spec_path: Path,
    reference_dir: Path,
    control_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    """JSON要求から参照済み3段IRを高さ変換し、最終SMFまで通す。"""

    inputs = _validated_inputs(Path(input_spec_path))
    raw_request = _read_object(Path(request_path), "composition request")
    normalized = normalize_composition_request(
        raw_request, published_controls={"高さ"}
    )
    reference_manifest = _read_object(Path(reference_dir) / "manifest.json", "reference manifest")
    reference_summary = _read_object(Path(reference_dir) / "summary.json", "reference summary")
    reference_hashes = reference_manifest.get("inputs")
    if not isinstance(reference_hashes, Mapping):
        raise HeightIntegrationError("reference manifest inputs are invalid")
    resolved = resolve_composition_request(
        normalized,
        reference_summary=reference_summary,
        reference_hashes=reference_hashes,
    )
    target = build_reference_generation_target(
        resolved.value,
        reference_dir=Path(reference_dir),
        control_dir=Path(control_dir),
    )
    baseline_reference = inputs["baseline_resolved_request"]["reference"]
    current_reference = resolved.value["reference"]
    if (
        baseline_reference.get("name") != current_reference.get("name")
        or baseline_reference.get("sha256") != current_reference.get("sha256")
    ):
        raise HeightIntegrationError(
            "request reference does not match the saved baseline provenance"
        )
    height_target = target.artifact["controls"]["height"]
    requested_height = float(height_target["value"])

    paths = inputs["artifact_paths"]
    plan = parse_piece_plan(paths["piece_plan"].read_text(encoding="utf-8"))
    score = parse_score_spec(paths["score_spec"].read_text(encoding="utf-8"))
    performance = parse_performance_spec(
        paths["performance_spec"].read_text(encoding="utf-8")
    )
    validate_pipeline(plan, score, performance)
    baseline_rendered = render_performance(plan, score, performance)
    baseline_quality = evaluate_generic_pipeline_quality(
        plan, score, performance, baseline_rendered
    )
    if not baseline_quality["passes"]:
        raise HeightIntegrationError(
            f"baseline generic quality failed: {baseline_quality['failures']}"
        )
    control_summary = _read_object(Path(control_dir) / "summary.json", "control summary")
    axes = control_summary.get("axes")
    if not isinstance(axes, Mapping) or not isinstance(axes.get("高さ"), Mapping):
        raise HeightIntegrationError("control summary height axis is invalid")
    axis = axes["高さ"]
    baseline_controls = _controls("baseline.mid", baseline_rendered, axes)

    implementation_hashes = _implementation_hashes()
    semantic_inputs = {
        "normalized_request": normalized.value,
        "normalized_request_sha256": normalized.sha256,
        "resolved_request": resolved.value,
        "resolved_request_sha256": resolved.sha256,
        "reference_target_sha256": target.sha256,
        "input_spec": inputs["spec"],
        "input_hashes": {
            "run_spec": inputs["run_spec_hash"],
            "manifest": inputs["manifest_hash"],
            "artifacts": inputs["artifact_hashes"],
            "baseline_resolved_request": inputs["baseline_resolved_request_hash"],
            "reference_manifest": sha256_file(Path(reference_dir) / "manifest.json"),
            "reference_summary": sha256_file(Path(reference_dir) / "summary.json"),
            "control_manifest": sha256_file(Path(control_dir) / "manifest.json"),
            "control_summary": sha256_file(Path(control_dir) / "summary.json"),
        },
    }
    run_id = build_run_id(semantic_inputs, implementation_hashes)
    run_dir = Path(output_root) / run_id
    reused = _verify_complete_run(run_dir)
    if reused is not None:
        return reused
    run_dir.mkdir(parents=True)

    output_paths = {
        "piece_plan": Path("outputs/piece-plan.dsl"),
        "score_spec": Path("outputs/score-spec.dsl"),
        "performance_spec": Path("outputs/performance-spec.dsl"),
        "candidates": Path("candidates.json"),
        "result": Path("result.json"),
    }
    observations: list[dict[str, Any]] = []
    accepted: dict[
        int,
        tuple[Any, RenderedPerformance, dict[str, Any], dict[str, Any], dict[str, Any]],
    ] = {}
    for semitones in candidate_transpositions(score):
        row: dict[str, Any] = {"semitones": semitones, "accepted": False}
        try:
            candidate = build_height_candidate(
                plan, score, performance, semitones=semitones
            )
            contract = validate_height_candidate_contract(
                plan, score, performance, candidate
            )
            validate_pipeline(candidate.plan, candidate.score, candidate.performance)
            rendered = render_performance(
                candidate.plan, candidate.score, candidate.performance
            )
            quality = evaluate_generic_pipeline_quality(
                candidate.plan, candidate.score, candidate.performance, rendered
            )
            controls = _controls(f"height-{semitones}.mid", rendered, axes)
            smf_relative = Path("candidates") / f"height-{semitones:+d}.mid"
            smf_path = run_dir / smf_relative
            render_performance_smf(rendered, smf_path)
            round_trip = _portable_round_trip(validate_smf_round_trip(rendered, smf_path))
            attack_unchanged = (
                controls["raw"]["発音頻度"]
                == baseline_controls["raw"]["発音頻度"]
            )
            accepted_flag = (
                contract["status"] == "passed"
                and quality["passes"]
                and round_trip["status"] == "passed"
                and attack_unchanged
            )
            descriptor = height_descriptor(rendered)
            row.update(
                {
                    "accepted": accepted_flag,
                    "height_mean": descriptor["mean"],
                    "height": descriptor,
                    "controls": controls,
                    "field_contract": contract,
                    "quality": quality,
                    "smf_round_trip": round_trip,
                    "attack_frequency_unchanged": attack_unchanged,
                    "artifact": smf_relative.as_posix(),
                    "artifact_sha256": sha256_file(smf_path),
                    "revoicing": asdict(candidate.diagnostics),
                }
            )
            rejection_reasons: list[str] = []
            if contract["status"] != "passed":
                rejection_reasons.append("height field contract failed")
            if not quality["passes"]:
                rejection_reasons.append("generic pipeline quality failed")
            if round_trip["status"] != "passed":
                rejection_reasons.append("SMF round-trip failed")
            if not attack_unchanged:
                rejection_reasons.append("attack frequency changed")
            if rejection_reasons:
                row["rejection_reasons"] = rejection_reasons
            if accepted_flag:
                accepted[semitones] = (
                    candidate,
                    rendered,
                    quality,
                    controls,
                    round_trip,
                )
        except (ValueError, HeightIntegrationError) as error:
            row["failure"] = str(error)
            row["rejection_reasons"] = [str(error)]
        observations.append(row)

    atomic_write_json(run_dir / output_paths["candidates"], observations)
    try:
        resolution = resolve_accepted_height(
            requested_height,
            observations=observations,
            minimum=float(axis["minimum"]),
            maximum=float(axis["maximum"]),
        )
    except ValueError as error:
        minimum = float(axis["minimum"])
        maximum = float(axis["maximum"])
        candidate_rejections = []
        for row in observations:
            reasons = list(row.get("rejection_reasons", []))
            if (
                row.get("accepted") is True
                and "height_mean" in row
                and not minimum <= float(row["height_mean"]) <= maximum
            ):
                reasons.append("height mean is outside the corpus range")
            candidate_rejections.append(
                {
                    "semitones": int(row["semitones"]),
                    "reasons": reasons or ["candidate was not selectable"],
                }
            )
        result = {
            "schema_version": 1,
            "scheme_id": SCHEME_ID,
            "status": "failed",
            "run_id": run_id,
            "reference": target.artifact["reference"],
            "requested_controls": target.artifact["controls"],
            "candidate_count": len(observations),
            "accepted_candidate_count": len(accepted),
            "candidate_rejections": candidate_rejections,
            "error": {
                "type": "HeightIntegrationError",
                "detail": str(error),
            },
            "artifacts": {
                "candidates": output_paths["candidates"].as_posix(),
                "result": output_paths["result"].as_posix(),
            },
            "provenance": {
                "semantic_inputs_sha256": sha256_json(semantic_inputs),
                "implementation_hashes": implementation_hashes,
            },
        }
        atomic_write_json(run_dir / output_paths["result"], result)
        manifest = {
            "schema_version": 1,
            "scheme_id": SCHEME_ID,
            "status": "failed",
            "run_id": run_id,
            "error": result["error"],
            "outputs": {
                output_paths["candidates"].as_posix(): sha256_file(
                    run_dir / output_paths["candidates"]
                ),
                output_paths["result"].as_posix(): sha256_file(
                    run_dir / output_paths["result"]
                ),
            },
        }
        atomic_write_json(run_dir / "manifest.json", manifest)
        return {**result, "run_dir": str(run_dir.resolve())}

    candidate, rendered, quality, controls, selected_round_trip = accepted[
        resolution.semitones
    ]
    achieved = resolution.status in {"exact", "quantized"}
    artifact_role = "normal_candidate" if achieved else "diagnostic_only"
    if achieved:
        output_paths["final_smf"] = Path("outputs/final.mid")
    atomic_write_bytes(
        run_dir / output_paths["piece_plan"],
        dump_piece_plan(candidate.plan).encode("utf-8"),
    )
    atomic_write_bytes(
        run_dir / output_paths["score_spec"],
        dump_score_spec(candidate.score).encode("utf-8"),
    )
    atomic_write_bytes(
        run_dir / output_paths["performance_spec"],
        dump_performance_spec(candidate.performance).encode("utf-8"),
    )
    final_round_trip = selected_round_trip
    if achieved:
        render_performance_smf(rendered, run_dir / output_paths["final_smf"])
        final_round_trip = _portable_round_trip(
            validate_smf_round_trip(rendered, run_dir / output_paths["final_smf"])
        )
    result = {
        "schema_version": 1,
        "scheme_id": SCHEME_ID,
        "status": "achieved" if achieved else "unreachable",
        "artifact_role": artifact_role,
        "run_id": run_id,
        "reference": target.artifact["reference"],
        "requested_controls": target.artifact["controls"],
        "resolution": asdict(resolution),
        "height": height_descriptor(rendered),
        "controls": controls,
        "quality": quality,
        "smf_round_trip": final_round_trip,
        "candidate_count": len(observations),
        "accepted_candidate_count": len(accepted),
        "artifacts": {key: path.as_posix() for key, path in output_paths.items()},
        "provenance": {
            "semantic_inputs_sha256": sha256_json(semantic_inputs),
            "implementation_hashes": implementation_hashes,
        },
    }
    atomic_write_json(run_dir / output_paths["result"], result)
    output_hashes = {
        path.as_posix(): sha256_file(run_dir / path)
        for path in output_paths.values()
    }
    output_hashes.update(
        {
            str(row["artifact"]): str(row["artifact_sha256"])
            for row in observations
            if "artifact" in row and "artifact_sha256" in row
        }
    )
    manifest = {
        "schema_version": 1,
        "scheme_id": SCHEME_ID,
        "status": "complete",
        "run_id": run_id,
        "outputs": output_hashes,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    return {**result, "run_dir": str(run_dir.resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--input-spec", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_height_integration(
        request_path=args.request,
        input_spec_path=args.input_spec,
        reference_dir=args.reference_dir,
        control_dir=args.control_dir,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
                "run_dir": result["run_dir"],
                "semitones": result.get("resolution", {}).get("semitones"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "achieved" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
