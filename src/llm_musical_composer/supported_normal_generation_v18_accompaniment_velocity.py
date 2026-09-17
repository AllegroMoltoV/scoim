"""V17の固定楽譜へ声部別velocity方針だけを適用して外部0回で再生する。"""

# ruff: noqa: E501 -- 固定証跡のパスとSHA-256を同じ行で対応付ける。

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .generic_pipeline_quality import (
    evaluate_generic_pipeline_quality,
    evaluate_generic_score_quality,
)
from .performance_pipeline import (
    analyze_voice_velocity,
    render_musicxml,
    render_performance,
    render_performance_smf,
)
from .pipeline_dsl import (
    dump_performance_spec,
    dump_score_spec,
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from .run_state import (
    RunStore,
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    sha256_json,
    sha256_text,
)
from .texture_budget import measure_rendered_texture_budget
from .whole_score_staged_generation_run import (
    calibrate_default_velocity,
    calibrate_key_release,
    evaluate_staged_candidate,
)


class SupportedNormalGenerationV18Error(RuntimeError):
    """V17の固定証跡またはvelocity限定変更が成立しない。"""


PROTOCOL_ID = "supported-normal-generation-v18-accompaniment-velocity"
SOURCE_ROOT_NAME = (
    ".appendix/supported-normal-generation-v17-nearest-key-release-unfit"
)
DEFAULT_ARTIFACT_NAME = (
    ".appendix/supported-normal-generation-v18-accompaniment-velocity"
)
VELOCITY_POLICY_ID = "foreground-accompaniment-harmony-shape-v1"
EXPECTED_SOURCE_SHA256 = {
    "manifest.json": "3781e992bae045b9cbcdb177db8243b08726e893c487a89e17381d7c899ff818",
    "runs/whole-score/run-state.json": "2b9c75953972193ed3e6433b175111508da2926fcfb295e319cd27eec07aa0e8",
    "runs/whole-score/run-spec.json": "5b0a298b319b656889d6ba61079fc717c3667e7c7acfb4387e7cab2573d35193",
    "runs/whole-score/inputs/piece-plan.dsl": "425b36623c78453cd462f5b9ef4795ef32e45745bf6ef44ba88af58d3db2e77d",
    "runs/whole-score/inputs/prompt-target.json": "90b24e40aee457dd6b1d2437428b64cabf0d027c6af0818b6428065a8f071da4",
    "runs/whole-score/inputs/texture-budget.json": "8906b61a03b6ad6a056364a1cfaf3e37c07f3d434be447f8c9f3e36b03d16341",
    "runs/whole-score/outputs/score-spec.dsl": "5e0e0913943f495bfc351d956cbd77d44c753984b38285f854ed5458368e429c",
    "runs/whole-score/outputs/performance-spec.dsl": "1d96891a4821a7fa8d0fa24f7cda6d238bbc7ac9cc8e1cc147420e72f7afad9d",
    "runs/whole-score/outputs/performance-collection.dsl": "9454e40fafaecf5be99e5c2146abeed0079eb24a7955a469404a0eff7abb8a6a",
    "runs/whole-score/staged/final.mid": "5334b74c0ce4e4113e0f21cc8069f189cb231596bdf211a4ebf71425bfda55c5",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupportedNormalGenerationV18Error(f"JSON object required: {path}")
    return value


def _write_text(path: Path, source: str) -> None:
    atomic_write_bytes(path, source.encode("utf-8"))


def _relative_record(project_root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(project_root).as_posix(),
        "sha256": sha256_file(path),
    }


def _semantic_target(prompt_target: dict[str, Any], target_id: str) -> dict[str, Any]:
    matches = [
        item
        for items in prompt_target.get("semantic_targets", {}).values()
        for item in items
        if isinstance(item, dict) and item.get("id") == target_id
    ]
    if len(matches) != 1:
        raise SupportedNormalGenerationV18Error(
            f"semantic target is missing: {target_id}"
        )
    return matches[0]


def _verify_source(project_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    source_root = project_root / SOURCE_ROOT_NAME
    for relative, expected in EXPECTED_SOURCE_SHA256.items():
        path = source_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SupportedNormalGenerationV18Error(
                f"source SHA-256 differs from reviewed evidence: {relative}"
            )
    manifest = _read_json(source_root / "manifest.json")
    if (
        manifest.get("status") != "completed_unfit"
        or manifest.get("promoted") is not False
        or manifest.get("confirmed_external_call_count") != 0
    ):
        raise SupportedNormalGenerationV18Error("V17 state differs from reviewed run")
    source_run = source_root / "runs/whole-score"
    source_spec = _read_json(source_run / "run-spec.json")
    return source_root, source_run, source_spec


def _initialize_run(
    project_root: Path,
    artifact_root: Path,
    source_root: Path,
    source_run: Path,
    source_spec: dict[str, Any],
) -> Path:
    run_dir = artifact_root / "runs/whole-score"
    model_config = dict(source_spec["model_config"])
    model_config["maximum_external_calls"] = 0
    generation_inputs = {
        relative: _relative_record(project_root, source_root / relative)
        for relative in EXPECTED_SOURCE_SHA256
    }
    spec = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_config": model_config,
        "input_hashes": {
            "generation_inputs": generation_inputs,
            "evaluation_inputs": source_spec["input_hashes"]["evaluation_inputs"],
        },
        "attack_frequency": source_spec["attack_frequency"],
        "register_enforcement": source_spec["register_enforcement"],
        "texture_placement_policy": source_spec["texture_placement_policy"],
        "velocity_policy_id": VELOCITY_POLICY_ID,
        "source_run_spec_sha256": sha256_file(source_run / "run-spec.json"),
    }
    store = RunStore(run_dir, max_calls=0)
    store.initialize(spec)
    for target, source in (
        ("inputs/piece-plan.dsl", "inputs/piece-plan.dsl"),
        ("inputs/prompt-target.json", "inputs/prompt-target.json"),
        ("inputs/texture-budget.json", "inputs/texture-budget.json"),
        ("inputs/source-score-spec.dsl", "outputs/score-spec.dsl"),
        ("inputs/source-performance-spec.dsl", "outputs/performance-spec.dsl"),
    ):
        store.snapshot_file(target, source_run / source)
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "staged",
            "passes": False,
            "promoted": False,
            "confirmed_external_call_count": 0,
        },
    )
    return run_dir


def _velocity_only_invariants(baseline: Any, candidate: Any) -> dict[str, bool]:
    baseline_note_shape = [
        (
            note.event_id,
            note.occurrence_node_id,
            note.at_ms,
            note.duration_ms,
            note.pitch,
            note.voice,
        )
        for note in baseline.notes
    ]
    candidate_note_shape = [
        (
            note.event_id,
            note.occurrence_node_id,
            note.at_ms,
            note.duration_ms,
            note.pitch,
            note.voice,
        )
        for note in candidate.notes
    ]
    return {
        "note_shape_equal": baseline_note_shape == candidate_note_shape,
        "pedals_equal": baseline.pedals == candidate.pedals,
        "harmonies_equal": baseline.harmonies == candidate.harmonies,
        "node_intervals_equal": baseline.node_intervals == candidate.node_intervals,
        "duration_equal": baseline.duration_ms == candidate.duration_ms,
        "velocity_changed": [note.velocity for note in baseline.notes]
        != [note.velocity for note in candidate.notes],
        "velocity_only_change": (
            baseline_note_shape == candidate_note_shape
            and baseline.pedals == candidate.pedals
            and baseline.harmonies == candidate.harmonies
            and baseline.node_intervals == candidate.node_intervals
            and baseline.duration_ms == candidate.duration_ms
            and [note.velocity for note in baseline.notes]
            != [note.velocity for note in candidate.notes]
        ),
    }


def run_supported_normal_generation_v18(
    project_root: Path,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """V17固定演奏を対照に、伴奏velocityだけを変更して再評価する。"""

    project_root = Path(project_root).resolve()
    artifact_root = (
        (project_root / DEFAULT_ARTIFACT_NAME).resolve()
        if artifact_root is None
        else Path(artifact_root).resolve()
    )
    if artifact_root.exists():
        raise SupportedNormalGenerationV18Error(
            f"V18 artifact root already exists: {artifact_root}"
        )
    source_root, source_run, source_spec = _verify_source(project_root)
    run_dir = _initialize_run(
        project_root, artifact_root, source_root, source_run, source_spec
    )
    plan = parse_piece_plan((run_dir / "inputs/piece-plan.dsl").read_text("utf-8"))
    score = parse_score_spec(
        (run_dir / "inputs/source-score-spec.dsl").read_text("utf-8")
    )
    baseline_performance = parse_performance_spec(
        (run_dir / "inputs/source-performance-spec.dsl").read_text("utf-8")
    )
    prompt_target = _read_json(run_dir / "inputs/prompt-target.json")
    texture_budget = _read_json(run_dir / "inputs/texture-budget.json")
    baseline = render_performance(plan, score, baseline_performance)
    performance = replace(
        baseline_performance,
        velocity_policy_id=VELOCITY_POLICY_ID,
    )
    performance, _, velocity_calibration = calibrate_default_velocity(
        plan,
        score,
        performance,
        _semantic_target(prompt_target, "velocity_shape"),
    )
    performance, rendered, key_release_calibration = calibrate_key_release(
        plan,
        score,
        performance,
        _semantic_target(prompt_target, "key_held_texture"),
        on_unreachable="nearest_unfit",
    )
    invariants = _velocity_only_invariants(baseline, rendered)
    if not invariants["velocity_only_change"]:
        raise SupportedNormalGenerationV18Error("V18 changed more than velocity")
    baseline_diagnostic = analyze_voice_velocity(plan, score, baseline_performance)
    candidate_diagnostic = analyze_voice_velocity(plan, score, performance)
    baseline_groups = baseline_diagnostic["accompaniment_attack_groups"]
    candidate_groups = candidate_diagnostic["accompaniment_attack_groups"]
    policy_applied = (
        candidate_groups["eligible_harmony_count"] > 0
        and candidate_groups["varied_eligible_harmony_count"] > 0
    )
    improved = (
        candidate_groups["adjacent_equal_ratio"]
        < baseline_groups["adjacent_equal_ratio"]
        or candidate_groups["longest_equal_run"]
        < baseline_groups["longest_equal_run"]
    )
    if not policy_applied or not improved:
        raise SupportedNormalGenerationV18Error(
            "V18 accompaniment velocity diagnostic did not improve"
        )
    diagnostic_comparison = {
        "schema_version": 1,
        "baseline": baseline_diagnostic,
        "candidate": candidate_diagnostic,
        "policy_applied": policy_applied,
        "improved": improved,
    }
    rendered_budget = measure_rendered_texture_budget(rendered, texture_budget)
    score_quality = evaluate_generic_score_quality(plan, score)
    pipeline_quality = evaluate_generic_pipeline_quality(
        plan, score, performance, rendered
    )
    generic_quality_passes = {
        "score": bool(score_quality["passes"]),
        "pipeline": bool(pipeline_quality["passes"]),
        "rendered_texture_budget": bool(
            rendered_budget["rendered_performance"]["matches_budget"]
        ),
    }
    if not all(generic_quality_passes.values()):
        raise SupportedNormalGenerationV18Error("V18 failed generic quality")
    score_source = dump_score_spec(score)
    performance_source = dump_performance_spec(performance)
    _write_text(run_dir / "outputs/score-spec.dsl", score_source)
    _write_text(run_dir / "outputs/performance-spec.dsl", performance_source)
    atomic_write_json(
        run_dir / "outputs/velocity-calibration.json", velocity_calibration
    )
    atomic_write_json(
        run_dir / "outputs/key-release-calibration.json", key_release_calibration
    )
    atomic_write_json(
        run_dir / "outputs/voice-velocity-diagnostic.json", diagnostic_comparison
    )
    atomic_write_json(
        run_dir / "outputs/rendered-texture-budget.json", rendered_budget
    )
    atomic_write_json(
        run_dir / "outputs/generic-quality.json",
        {"score": score_quality, "pipeline": pipeline_quality},
    )
    musicxml = render_musicxml(plan, score, run_dir / "staged/final.musicxml")
    smf = render_performance_smf(rendered, run_dir / "staged/final.mid").path
    RunStore(run_dir, max_calls=0).record_step(
        "performance-collection",
        "completed",
        {
            "source_score_spec": sha256_text(score_source),
            "source_performance_spec": EXPECTED_SOURCE_SHA256[
                "runs/whole-score/outputs/performance-spec.dsl"
            ],
            "prompt_target": sha256_json(prompt_target),
            "texture_budget": sha256_json(texture_budget),
        },
        {
            "source_mode": "prepared-v17-performance",
            "external_call_number": None,
            "performance_spec_sha256": sha256_text(performance_source),
            "musicxml_sha256": sha256_file(musicxml),
            "smf_sha256": sha256_file(smf),
            "velocity_calibration_sha256": sha256_json(velocity_calibration),
            "key_release_calibration_sha256": sha256_json(
                key_release_calibration
            ),
            "voice_velocity_diagnostic_sha256": sha256_json(
                diagnostic_comparison
            ),
        },
    )
    evaluation = evaluate_staged_candidate(project_root, run_dir)
    if evaluation.get("status") != "completed_unfit" or evaluation.get("promoted"):
        raise SupportedNormalGenerationV18Error(
            "V18 candidate evaluation changed reference-unfit status"
        )
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": evaluation["status"],
        "passes": bool(evaluation["passes"]),
        "promoted": bool(evaluation["promoted"]),
        "confirmed_external_call_count": 0,
        "generic_quality_passes": generic_quality_passes,
        "invariants": invariants,
        "velocity_diagnostic_comparison": {
            "policy_applied": policy_applied,
            "improved": improved,
        },
        "velocity_calibration": velocity_calibration,
        "key_release_calibration": key_release_calibration,
        "evaluation": evaluation,
        "smf": {"path": str(smf), "sha256": sha256_file(smf)},
        "musicxml": {"path": str(musicxml), "sha256": sha256_file(musicxml)},
    }
    atomic_write_json(artifact_root / "manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path, default=None)
    arguments = parser.parse_args()
    result = run_supported_normal_generation_v18(
        arguments.project_root,
        arguments.artifact_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "completed_unfit" else 1


if __name__ == "__main__":
    raise SystemExit(main())
