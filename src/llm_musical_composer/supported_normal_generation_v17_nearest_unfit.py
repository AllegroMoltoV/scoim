"""V16の固定応答を参照不適合候補として外部0回で完走する。"""

# ruff: noqa: E501 -- 固定証跡のパスとSHA-256を同じ行で対応付ける。

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .generic_pipeline_quality import (
    evaluate_generic_pipeline_quality,
    evaluate_generic_score_quality,
)
from .performance_pipeline import render_musicxml, render_performance_smf
from .pipeline_dsl import (
    dump_performance_spec,
    dump_score_spec,
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
from .whole_score_staged_generation import assemble_whole_score_performance
from .whole_score_staged_generation_dsl import (
    dump_performance_collection,
    parse_performance_collection,
)
from .whole_score_staged_generation_run import (
    calibrate_default_velocity,
    calibrate_key_release,
    evaluate_staged_candidate,
)


class SupportedNormalGenerationV17Error(RuntimeError):
    """V17の固定証跡または一般品質契約が成立しない。"""


PROTOCOL_ID = "supported-normal-generation-v17-nearest-key-release-unfit"
SOURCE_ROOT_NAME = (
    ".appendix/supported-normal-generation-v16-shared-ending-contract"
)
DEFAULT_ARTIFACT_NAME = (
    ".appendix/supported-normal-generation-v17-nearest-key-release-unfit"
)
EXPECTED_CANONICAL_PERFORMANCE_SHA256 = (
    "9454e40fafaecf5be99e5c2146abeed0079eb24a7955a469404a0eff7abb8a6a"
)
EXPECTED_SOURCE_SHA256 = {
    "manifest.json": "dc98febd99feee6e51fcaa171fd5c8501020a85ac4b2d7b3727a54717b65967d",
    "runs/whole-score/run-state.json": "119bdb1a1bb9aab60b2346eadc8dbc7e780ae7af87b547758b86ba5aac446d9f",
    "runs/whole-score/run-spec.json": "ad9c7cca301514da8ecd8c670c63a937a4f5d955374e71540475d5e94a96b60c",
    "runs/whole-score/inputs/piece-plan.dsl": "425b36623c78453cd462f5b9ef4795ef32e45745bf6ef44ba88af58d3db2e77d",
    "runs/whole-score/inputs/prompt-target.json": "90b24e40aee457dd6b1d2437428b64cabf0d027c6af0818b6428065a8f071da4",
    "runs/whole-score/inputs/prepared-texture-budget.json": "8906b61a03b6ad6a056364a1cfaf3e37c07f3d434be447f8c9f3e36b03d16341",
    "runs/whole-score/inputs/prepared-texture-prefix-provenance.json": "3646fc16f98b11390c52ffc822b0235be0cb91ee5286821da8fd89550b8cb12c",
    "runs/whole-score/outputs/score-spec.dsl": "5e0e0913943f495bfc351d956cbd77d44c753984b38285f854ed5458368e429c",
    "runs/whole-score/outputs/texture-collection.dsl": "10c928429ecc881bddbd231c5fd1c2d2620e6d4e8ba8570c344794c011f2f171",
    "runs/whole-score/outputs/generic-score-quality.json": "8aef1eb4ddcbae800273e24ca1418e9ff666bfdda88abe10669d05b54311944c",
    "runs/whole-score/steps/texture-collection-batch-001.json": "4cac88828e7fbd4fc5bee516373297a031fc48b299667ae7782261bed4c2867e",
    "runs/whole-score/outputs/texture-collection-batch-001.dsl": "c9bb578955fd64d21a4d41ccb08eb9d16ec73ab75078790e17ea37341fee3d4b",
    "runs/whole-score/steps/texture-collection-batch-002.json": "3b245608a1bd367f3dc5e2ecbaf62b7b24d19b2f89f84a09518300586594b55b",
    "runs/whole-score/outputs/texture-collection-batch-002.dsl": "f801c88e02cd7fe00a245360b527a408cec67e8f8921f6c11ddeba87faa86cd3",
    "runs/whole-score/steps/texture-collection-batch-003.json": "a042af3cdf8f0398ecbfcb209b99adf498e70cf773b625c1b44c5dffa44a04b8",
    "runs/whole-score/outputs/texture-collection-batch-003.dsl": "b1cb17a92c30433f30487609786b901248d2c9468ff65e2ee8a75d21458ea17b",
    "runs/whole-score/steps/texture-collection-batch-004.json": "129f4b03af1501979e5d74fd68a5926411e876454765bdd93c23d3f71351ea7c",
    "runs/whole-score/prompts/texture-collection-batch-004.md": "3cff9d62c52002db6bb082fb113d3486df86b89c7eb9f5a7b4776292d1dab312",
    "runs/whole-score/responses/texture-collection-batch-004.dsl": "e8405361792c7e1eef44984e830f4343afda0fabdf51b40ae27e6532c0bcca6d",
    "runs/whole-score/outputs/texture-collection-batch-004.dsl": "9814b3cf5580266889025c9c9e715bd8b3d401a0660ad184c7bd109c1e333100",
    "runs/whole-score/steps/performance-collection.json": "a5ed254ab9d08771fbf3ea96d483eb6db46d9ba11800126849847cd95f978a96",
    "runs/whole-score/prompts/performance-collection.md": "17cbd1bc9114c39c212070d8c30a709c521afd804ef5ed9e8b4b526394320d66",
    "runs/whole-score/responses/performance-collection.dsl": "d27b964122ac87890195c363edc94ade094bed8824dc2c54e02209fd7db6055b",
    "runs/whole-score/failures/performance-collection.json": "088936314828396f57adcc985e7827eaab035d5e6985ddda1920128b751ef564",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupportedNormalGenerationV17Error(f"JSON object required: {path}")
    return value


def _write_text(path: Path, source: str) -> None:
    atomic_write_bytes(path, source.encode("utf-8"))


def _relative_record(project_root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(project_root).as_posix(),
        "sha256": sha256_file(path),
    }


def _verify_file_records(project_root: Path, records: dict[str, Any]) -> None:
    for name, record in records.items():
        if not isinstance(record, dict):
            raise SupportedNormalGenerationV17Error(
                f"evaluation record is invalid: {name}"
            )
        path = project_root / str(record.get("path", ""))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise SupportedNormalGenerationV17Error(
                f"evaluation input SHA-256 differs: {name}"
            )


def _semantic_target(prompt_target: dict[str, Any], target_id: str) -> dict[str, Any]:
    matches = [
        item
        for items in prompt_target.get("semantic_targets", {}).values()
        for item in items
        if isinstance(item, dict) and item.get("id") == target_id
    ]
    if len(matches) != 1:
        raise SupportedNormalGenerationV17Error(
            f"semantic target is missing: {target_id}"
        )
    return matches[0]


def _verify_source(project_root: Path) -> dict[str, Any]:
    source_root = project_root / SOURCE_ROOT_NAME
    for relative, expected in EXPECTED_SOURCE_SHA256.items():
        path = source_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SupportedNormalGenerationV17Error(
                f"source SHA-256 differs from reviewed evidence: {relative}"
            )
    source_run = source_root / "runs/whole-score"
    manifest = _read_json(source_root / "manifest.json")
    state = _read_json(source_run / "run-state.json")
    failure = _read_json(source_run / "failures/performance-collection.json")
    calls = state.get("calls", {})
    performance_step = state.get("steps", {}).get("performance-collection", {})
    if (
        manifest.get("status") != "failed"
        or state.get("status") != "failed"
        or calls.get("confirmed_external_call_count") != 2
        or calls.get("successful_external_call_count") != 2
        or performance_step.get("status") != "failed"
        or failure.get("detail")
        != "key release calibration has no passing integer percent"
    ):
        raise SupportedNormalGenerationV17Error(
            "V16 state differs from reviewed calibration failure"
        )
    if any(
        path.exists()
        for path in (
            source_run / "staged/final.mid",
            source_run / "outputs/final.mid",
            source_run / "outputs/performance-collection.dsl",
        )
    ):
        raise SupportedNormalGenerationV17Error(
            "V16 unexpectedly contains post-calibration artifacts"
        )
    for index in range(1, 4):
        step = state["steps"][f"texture-collection-batch-{index:03d}"]
        if (
            step.get("status") != "completed"
            or step.get("outputs", {}).get("source_mode") != "prepared"
            or step.get("outputs", {}).get("external_call_number") is not None
        ):
            raise SupportedNormalGenerationV17Error(
                "V16 prepared texture provenance differs"
            )
    fourth = state["steps"]["texture-collection-batch-004"]
    if (
        fourth.get("status") != "completed"
        or fourth.get("outputs", {}).get("source_mode") != "external"
        or fourth.get("outputs", {}).get("external_call_number") != 1
    ):
        raise SupportedNormalGenerationV17Error(
            "V16 external texture provenance differs"
        )
    raw_performance = (
        source_run / "responses/performance-collection.dsl"
    ).read_text(encoding="utf-8")
    drafts = parse_performance_collection(raw_performance)
    canonical = dump_performance_collection(drafts)
    if sha256_text(canonical) != EXPECTED_CANONICAL_PERFORMANCE_SHA256:
        raise SupportedNormalGenerationV17Error(
            "derived PerformanceCollection SHA-256 differs"
        )
    source_spec = _read_json(source_run / "run-spec.json")
    evaluation_inputs = source_spec.get("input_hashes", {}).get(
        "evaluation_inputs"
    )
    if not isinstance(evaluation_inputs, dict):
        raise SupportedNormalGenerationV17Error("V16 evaluation inputs are missing")
    _verify_file_records(project_root, evaluation_inputs)
    return {
        "source_root": source_root,
        "source_run": source_run,
        "source_spec": source_spec,
        "evaluation_inputs": evaluation_inputs,
        "performance_drafts": drafts,
        "canonical_performance": canonical,
    }


def _initialize_run(
    project_root: Path,
    artifact_root: Path,
    verified: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    source_run = verified["source_run"]
    run_dir = artifact_root / "runs/whole-score"
    source_spec = verified["source_spec"]
    generation_inputs = {
        relative: _relative_record(
            project_root, verified["source_root"] / relative
        )
        for relative in EXPECTED_SOURCE_SHA256
    }
    model_config = dict(source_spec["model_config"])
    model_config["maximum_external_calls"] = 0
    spec = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_config": model_config,
        "input_hashes": {
            "generation_inputs": generation_inputs,
            "evaluation_inputs": verified["evaluation_inputs"],
        },
        "attack_frequency": source_spec["attack_frequency"],
        "register_enforcement": source_spec["register_enforcement"],
        "texture_placement_policy": source_spec["texture_placement_policy"],
        "source_run_spec_sha256": sha256_file(source_run / "run-spec.json"),
    }
    store = RunStore(run_dir, max_calls=0)
    store.initialize(spec)
    store.snapshot_file(
        "inputs/piece-plan.dsl", source_run / "inputs/piece-plan.dsl"
    )
    store.snapshot_file(
        "inputs/prompt-target.json", source_run / "inputs/prompt-target.json"
    )
    store.snapshot_file(
        "inputs/source-score-spec.dsl", source_run / "outputs/score-spec.dsl"
    )
    store.snapshot_file(
        "inputs/texture-budget.json",
        source_run / "inputs/prepared-texture-budget.json",
    )
    store.snapshot_file(
        "inputs/source-performance-collection.dsl",
        source_run / "responses/performance-collection.dsl",
    )
    provenance = {
        "schema_version": 1,
        "status": "verified",
        "source_artifact": verified["source_root"].relative_to(
            project_root
        ).as_posix(),
        "source_files": generation_inputs,
        "derived_canonical_performance_sha256": (
            EXPECTED_CANONICAL_PERFORMANCE_SHA256
        ),
        "confirmed_external_call_count": 2,
        "v17_external_call_count": 0,
    }
    store.snapshot_json("inputs/source-provenance.json", provenance)
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
    return run_dir, provenance


def run_supported_normal_generation_v17(
    project_root: Path,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """V16固定応答を一度だけ検証し、参照不適合SMFまで保存する。"""

    project_root = Path(project_root).resolve()
    artifact_root = (
        (project_root / DEFAULT_ARTIFACT_NAME).resolve()
        if artifact_root is None
        else Path(artifact_root).resolve()
    )
    if artifact_root.exists():
        raise SupportedNormalGenerationV17Error(
            f"V17 artifact root already exists: {artifact_root}"
        )
    verified = _verify_source(project_root)
    run_dir, provenance = _initialize_run(project_root, artifact_root, verified)
    plan = parse_piece_plan(
        (run_dir / "inputs/piece-plan.dsl").read_text(encoding="utf-8")
    )
    score = parse_score_spec(
        (run_dir / "inputs/source-score-spec.dsl").read_text(encoding="utf-8")
    )
    score_source = dump_score_spec(score)
    if sha256_text(score_source) != EXPECTED_SOURCE_SHA256[
        "runs/whole-score/outputs/score-spec.dsl"
    ]:
        raise SupportedNormalGenerationV17Error("source ScoreSpec is not canonical")
    prompt_target = _read_json(run_dir / "inputs/prompt-target.json")
    texture_budget = _read_json(run_dir / "inputs/texture-budget.json")
    performance = assemble_whole_score_performance(
        "supported-normal-generation-v17",
        plan,
        score,
        verified["performance_drafts"],
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
    if (
        key_release_calibration.get("status") != "nearest_unfit"
        or key_release_calibration.get("selected_percent") != 59
    ):
        raise SupportedNormalGenerationV17Error(
            "V17 key release counterfactual differs"
        )
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
        raise SupportedNormalGenerationV17Error(
            "V17 fixed performance failed generic quality"
        )
    _write_text(run_dir / "outputs/score-spec.dsl", score_source)
    _write_text(
        run_dir / "outputs/performance-collection.dsl",
        verified["canonical_performance"],
    )
    performance_source = dump_performance_spec(performance)
    _write_text(run_dir / "outputs/performance-spec.dsl", performance_source)
    atomic_write_json(
        run_dir / "outputs/velocity-calibration.json", velocity_calibration
    )
    atomic_write_json(
        run_dir / "outputs/key-release-calibration.json",
        key_release_calibration,
    )
    atomic_write_json(
        run_dir / "outputs/rendered-texture-budget.json", rendered_budget
    )
    atomic_write_json(
        run_dir / "outputs/generic-quality.json",
        {"score": score_quality, "pipeline": pipeline_quality},
    )
    staged_dir = run_dir / "staged"
    musicxml = render_musicxml(plan, score, staged_dir / "final.musicxml")
    smf = render_performance_smf(rendered, staged_dir / "final.mid").path
    RunStore(run_dir, max_calls=0).record_step(
        "performance-collection",
        "completed",
        {
            "source_score_spec": sha256_text(score_source),
            "source_performance_collection": EXPECTED_SOURCE_SHA256[
                "runs/whole-score/responses/performance-collection.dsl"
            ],
            "prompt_target": sha256_json(prompt_target),
            "texture_budget": sha256_json(texture_budget),
        },
        {
            "source_mode": "prepared-v16-response",
            "external_call_number": None,
            "performance_spec_sha256": sha256_text(performance_source),
            "musicxml_sha256": sha256_file(musicxml),
            "smf_sha256": sha256_file(smf),
            "velocity_calibration_sha256": sha256_json(velocity_calibration),
            "key_release_calibration_sha256": sha256_json(
                key_release_calibration
            ),
        },
    )
    evaluation = evaluate_staged_candidate(project_root, run_dir)
    reference_fit = {
        "key_held_texture": bool(
            evaluation["semantic_texture"]["key_held_texture"][
                "within_neighborhood"
            ]
        ),
        "texture_fit": bool(evaluation["texture_fit"]),
    }
    if (
        evaluation.get("status") != "completed_unfit"
        or evaluation.get("promoted")
        or reference_fit["key_held_texture"]
        or reference_fit["texture_fit"]
    ):
        raise SupportedNormalGenerationV17Error(
            "V17 candidate evaluation did not preserve reference unfit status"
        )
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": evaluation["status"],
        "passes": bool(evaluation["passes"]),
        "promoted": bool(evaluation["promoted"]),
        "confirmed_external_call_count": 0,
        "source_provenance_sha256": sha256_json(provenance),
        "key_release_calibration": key_release_calibration,
        "generic_quality_passes": generic_quality_passes,
        "reference_fit": reference_fit,
        "evaluation": evaluation,
        "smf": {"path": str(smf), "sha256": sha256_file(smf)},
        "musicxml": {
            "path": str(musicxml),
            "sha256": sha256_file(musicxml),
        },
    }
    atomic_write_json(artifact_root / "manifest.json", result)
    return result
