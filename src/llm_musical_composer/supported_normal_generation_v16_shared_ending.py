"""V15bの検証済み前半を再利用し、共同終止から最終SMFまで実走する。"""

# ruff: noqa: E501 -- 固定証跡のパスとSHA-256を同じ行で対応付ける。

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .pipeline_dsl import parse_piece_plan
from .run_state import atomic_write_json, sha256_file, sha256_json
from .texture_budget import allocate_texture_budget
from .whole_score_staged_generation import (
    assemble_whole_score_melodies,
    assemble_whole_score_skeleton,
)
from .whole_score_staged_generation_dsl import (
    dump_melody_collection,
    dump_texture_collection,
    parse_harmonic_collection,
    parse_melody_collection,
    parse_texture_collection,
)
from .whole_score_staged_generation_run import (
    WholeScorePreparedSource,
    _validate_partial_texture_batches,
    create_default_whole_score_runner,
    execute_prepared_whole_score_live_run,
    normalize_melody_ending,
    partition_texture_batches,
    prepare_whole_score_live_run,
)


class SupportedNormalGenerationV16Error(RuntimeError):
    """V16の固定入力または再利用契約が確認済み証跡と異なる。"""


SOURCE_ROOT_NAME = (
    ".appendix/supported-normal-generation-v15b-bidirectional-low-spacing-v6-network-access"
)
SOURCE_RUN_NAME = (
    ".appendix/supported-normal-generation-v10-velocity-register-"
    "transition-contract/runs/piece-plan"
)
DEFAULT_ARTIFACT_NAME = ".appendix/supported-normal-generation-v16-shared-ending-contract"
ADOPTED_MATERIAL_IDS = (
    "material_1",
    "material_2",
    "material_3",
    "material_4",
    "material_5",
)
EXCLUDED_MATERIAL_IDS = ("material_6",)
BATCH_MAXIMUM_EVENT_COUNT = 400
MAXIMUM_EXTERNAL_CALLS = 2

EXPECTED_SOURCE_SHA256 = {
    "manifest.json": "9f8212245d878a6f4da362493545c44584a8c567a3338e04097720267da096b3",
    "runs/whole-score/run-state.json": "3a44f8490f74b20c713f261b7820f8a9842bf643610698b604a523d5a343c874",
    "runs/whole-score/run-spec.json": "adc491e8943e4b060388dc4ebd92d39433a83a45a65b3134fc8a08a42e4ac3ad",
    "runs/whole-score/inputs/piece-plan.dsl": "425b36623c78453cd462f5b9ef4795ef32e45745bf6ef44ba88af58d3db2e77d",
    "runs/whole-score/inputs/prompt-target.json": "90b24e40aee457dd6b1d2437428b64cabf0d027c6af0818b6428065a8f071da4",
    "runs/whole-score/inputs/resolved-request.json": "baf29a0c1d10558fff762a22a7b42c677196560b4ba8480f26671049b2d6fed7",
    "runs/whole-score/inputs/prepared-harmonic-collection.dsl": "781dfa6ebc7834c012d4ff0fe84785f867dc52c03f25b9b6d5eed97b0ba345a5",
    "runs/whole-score/inputs/prepared-melody-collection.dsl": "6c8b5d0987d7966e2beb45a0837ed46e7b64fe23d6c7c3b041b2a7ce5a2469fb",
    "runs/whole-score/inputs/prepared-texture-budget.json": "8906b61a03b6ad6a056364a1cfaf3e37c07f3d434be447f8c9f3e36b03d16341",
    "runs/whole-score/steps/texture-collection-batch-001.json": "eb25bacfaf74629c82cf317ba74ea454370d28fbff3f2e7d637e1b012db3f95d",
    "runs/whole-score/prompts/texture-collection-batch-001.md": "9478fd266e6cec3d9fcb94e6789168989894fafa271615fab0b5c04b55f016ba",
    "runs/whole-score/responses/texture-collection-batch-001.dsl": "f633eb666a84cb3775f124215a2203075b964b925dcd40f08144fda078789e37",
    "runs/whole-score/outputs/texture-collection-batch-001.dsl": "c9bb578955fd64d21a4d41ccb08eb9d16ec73ab75078790e17ea37341fee3d4b",
    "runs/whole-score/steps/texture-collection-batch-002.json": "dd3c2791084ff1758673d3a639c7354777b5564dca8d6dacf265d36fcaa11340",
    "runs/whole-score/prompts/texture-collection-batch-002.md": "83b0239d5273009594ebcdd3d632bfa278ac12d2e55ff56178953617307106cf",
    "runs/whole-score/responses/texture-collection-batch-002.dsl": "a7ee31ee5b5f73eed51e265fd6d6ee60d1f6584c7c8424d1c491c2e41d36156c",
    "runs/whole-score/outputs/texture-collection-batch-002.dsl": "f801c88e02cd7fe00a245360b527a408cec67e8f8921f6c11ddeba87faa86cd3",
    "runs/whole-score/steps/texture-collection-batch-003.json": "eef275c5b501b174c144eb7f8b698a95d16c4c329857156f575d2526a3c2e59f",
    "runs/whole-score/prompts/texture-collection-batch-003.md": "a7a48e17e3f95077dc18829bad9c6f4d2256a8cd00d4ec226cd844879cf23082",
    "runs/whole-score/responses/texture-collection-batch-003.dsl": "025942619f347d9972fcd5217d7348fc18ac95ddcee392ab2135584d8f411dfe",
    "runs/whole-score/outputs/texture-collection-batch-003.dsl": "b1cb17a92c30433f30487609786b901248d2c9468ff65e2ee8a75d21458ea17b",
    "runs/whole-score/steps/texture-collection-batch-004.json": "bec3b7f2e5a7818052519bfe97fbff8429d218bea59c9055219f34e4236a374e",
    "runs/whole-score/prompts/texture-collection-batch-004.md": "86f5166aec7191a763e19dc85b0de8404c975390f0f1b93a5be90dbad8e108dc",
    "runs/whole-score/responses/texture-collection-batch-004.dsl": "abb6faac958e1e6ec1ab2eeb798a322b39204ec1128d6705e95cf0442de0b723",
    "runs/whole-score/outputs/texture-collection-batch-004.dsl": "71adf47d6a3c9ccfa6e98aa0231587192772f9879c61f1f176e922feafb5b4b0",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupportedNormalGenerationV16Error(f"JSON is not an object: {path}")
    return value


def _relative_record(project_root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(project_root).as_posix(),
        "sha256": sha256_file(path),
    }


def _verify_source(project_root: Path) -> tuple[Path, dict[str, Any]]:
    source_root = project_root / SOURCE_ROOT_NAME
    for relative, expected in EXPECTED_SOURCE_SHA256.items():
        path = source_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SupportedNormalGenerationV16Error(
                f"source SHA-256 differs from reviewed evidence: {relative}"
            )
    manifest = _read_json(source_root / "manifest.json")
    state = _read_json(source_root / "runs/whole-score/run-state.json")
    calls = state.get("calls", {})
    if (
        manifest.get("status") != "failed"
        or state.get("status") != "failed"
        or calls.get("confirmed_external_call_count") != 4
        or calls.get("successful_external_call_count") != 4
        or "performance-collection" in state.get("steps", {})
    ):
        raise SupportedNormalGenerationV16Error("source run state differs from reviewed evidence")
    source_run = source_root / "runs/whole-score"
    if any(
        path.exists()
        for path in (
            source_run / "outputs/performance-spec.dsl",
            source_run / "outputs/final.mid",
            source_run / "staged/final.mid",
        )
    ):
        raise SupportedNormalGenerationV16Error("source unexpectedly contains a performance or SMF")
    expected_batches = (
        ("material_1", "material_2"),
        ("material_3",),
        ("material_4", "material_5"),
        ("material_6",),
    )
    for index, expected_ids in enumerate(expected_batches, 1):
        suffix = f"{index:03d}"
        step = state["steps"].get(f"texture-collection-batch-{suffix}", {})
        prompt = source_run / f"prompts/texture-collection-batch-{suffix}.md"
        output = source_run / f"outputs/texture-collection-batch-{suffix}.dsl"
        if (
            step.get("status") != "completed"
            or tuple(step.get("outputs", {}).get("material_ids", ())) != expected_ids
            or step.get("outputs", {}).get("external_call_number") != index
            or step.get("input_hashes", {}).get("prompt") != sha256_file(prompt)
            or step.get("outputs", {}).get("texture_draft_sha256") != sha256_file(output)
        ):
            raise SupportedNormalGenerationV16Error(f"source batch {suffix} provenance differs")
    return source_root, state


def validate_adopted_melody_projections(
    source: tuple[Any, ...],
    normalized: tuple[Any, ...],
    diagnostic: dict[str, Any],
    *,
    adopted_material_ids: tuple[str, ...],
) -> dict[str, Any]:
    """採用素材の旋律が終止正規化の影響を受けていないことを検査する。"""

    if len(source) != len(normalized) or len(adopted_material_ids) >= len(source):
        raise SupportedNormalGenerationV16Error("adopted melody projection is invalid")
    adopted_count = len(adopted_material_ids)
    if source[:adopted_count] != normalized[:adopted_count]:
        raise SupportedNormalGenerationV16Error("adopted melody projection differs")
    if diagnostic.get("material_id") in adopted_material_ids:
        raise SupportedNormalGenerationV16Error("adopted melody normalization differs")
    return {
        "schema_version": 1,
        "adopted_material_ids": list(adopted_material_ids),
        "source_projection_sha256": sha256_json([asdict(item) for item in source[:adopted_count]]),
        "normalized_projection_sha256": sha256_json(
            [asdict(item) for item in normalized[:adopted_count]]
        ),
        "matches": True,
    }


def _attack_target(prompt_target: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for item in prompt_target.get("semantic_targets", {}).get("score_spec", [])
        if isinstance(item, dict) and item.get("id") == "attack_texture"
    ]
    if len(matches) != 1:
        raise SupportedNormalGenerationV16Error("attack_texture target is missing")
    return matches[0]


def _build_preflight(project_root: Path) -> dict[str, Any]:
    source_root, state = _verify_source(project_root)
    source_run = source_root / "runs/whole-score"
    plan = parse_piece_plan((source_run / "inputs/piece-plan.dsl").read_text(encoding="utf-8"))
    harmonies = parse_harmonic_collection(
        (source_run / "inputs/prepared-harmonic-collection.dsl").read_text(encoding="utf-8")
    )
    source_melodies = parse_melody_collection(
        (source_run / "inputs/prepared-melody-collection.dsl").read_text(encoding="utf-8")
    )
    skeleton = assemble_whole_score_skeleton(
        "v16-preflight",
        plan,
        tuple((item.length_units, item.draft) for item in harmonies),
    )
    normalized_melodies, normalization = normalize_melody_ending(plan, skeleton, source_melodies)
    if normalization.get("status") != "normalized":
        raise SupportedNormalGenerationV16Error(
            "source melody no longer reproduces the shared-ending counterfactual"
        )
    projection = validate_adopted_melody_projections(
        source_melodies,
        normalized_melodies,
        normalization,
        adopted_material_ids=ADOPTED_MATERIAL_IDS,
    )
    melodies = assemble_whole_score_melodies("v16-preflight", plan, skeleton, normalized_melodies)
    saved_budget = _read_json(source_run / "inputs/prepared-texture-budget.json")
    rebuilt_budget = allocate_texture_budget(
        plan,
        skeleton,
        melodies,
        _attack_target(_read_json(source_run / "inputs/prompt-target.json")),
        minimum_attack_group_count=int(saved_budget["score_spec_target"]["attack_group_count"]),
    )
    if rebuilt_budget != saved_budget:
        raise SupportedNormalGenerationV16Error(
            "normalized melody changes the prepared texture budget"
        )
    batches = partition_texture_batches(
        saved_budget["materials"], maximum_event_count=BATCH_MAXIMUM_EVENT_COUNT
    )
    expected_batches = [
        ["material_1", "material_2"],
        ["material_3"],
        ["material_4", "material_5"],
        ["material_6"],
    ]
    if [[item["material_id"] for item in batch] for batch in batches] != expected_batches:
        raise SupportedNormalGenerationV16Error("texture batch partition differs")
    adopted_drafts = []
    source_records = []
    for index in range(1, 4):
        suffix = f"{index:03d}"
        output_path = source_run / f"outputs/texture-collection-batch-{suffix}.dsl"
        adopted_drafts.extend(parse_texture_collection(output_path.read_text(encoding="utf-8")))
        source_records.append(
            {
                name: _relative_record(project_root, source_run / relative)
                for name, relative in {
                    "step": f"steps/texture-collection-batch-{suffix}.json",
                    "prompt": f"prompts/texture-collection-batch-{suffix}.md",
                    "raw_response": f"responses/texture-collection-batch-{suffix}.dsl",
                    "canonical_output": f"outputs/texture-collection-batch-{suffix}.dsl",
                }.items()
            }
        )
    if len(adopted_drafts) != len(ADOPTED_MATERIAL_IDS):
        raise SupportedNormalGenerationV16Error("adopted texture prefix count differs")
    drafts_by_material = dict(zip(ADOPTED_MATERIAL_IDS, adopted_drafts, strict=True))
    validation = _validate_partial_texture_batches(
        plan,
        skeleton,
        melodies,
        saved_budget,
        drafts_by_material,
        (38, 86),
        "bidirectional-low-spacing-v6",
    )
    if validation.get("whole_score_quality") is not None:
        raise SupportedNormalGenerationV16Error(
            "partial prefix unexpectedly ran whole-score quality"
        )
    excluded_suffix = "004"
    excluded = {
        name: _relative_record(project_root, source_run / relative)
        for name, relative in {
            "step": f"steps/texture-collection-batch-{excluded_suffix}.json",
            "prompt": f"prompts/texture-collection-batch-{excluded_suffix}.md",
            "raw_response": f"responses/texture-collection-batch-{excluded_suffix}.dsl",
            "canonical_output": f"outputs/texture-collection-batch-{excluded_suffix}.dsl",
        }.items()
    }
    provenance = {
        "schema_version": 1,
        "status": "verified",
        "source_manifest": _relative_record(project_root, source_root / "manifest.json"),
        "source_run_state": _relative_record(project_root, source_run / "run-state.json"),
        "source_inputs": {
            name: _relative_record(project_root, source_run / relative)
            for name, relative in {
                "piece_plan": "inputs/piece-plan.dsl",
                "prompt_target": "inputs/prompt-target.json",
                "resolved_request": "inputs/resolved-request.json",
                "harmonic_collection": "inputs/prepared-harmonic-collection.dsl",
                "melody_collection": "inputs/prepared-melody-collection.dsl",
                "texture_budget": "inputs/prepared-texture-budget.json",
            }.items()
        },
        "source_confirmed_external_call_count": state["calls"]["confirmed_external_call_count"],
        "adopted_batches": source_records,
        "adopted_material_ids": list(ADOPTED_MATERIAL_IDS),
        "excluded_batch": excluded,
        "excluded_material_ids": list(EXCLUDED_MATERIAL_IDS),
        "melody_projection": projection,
        "melody_ending_normalization": normalization,
        "prefix_validation_sha256": sha256_json(validation),
    }
    return {
        "source_root": source_root,
        "source_run": source_run,
        "harmonies": harmonies,
        "normalized_melodies": normalized_melodies,
        "texture_budget": saved_budget,
        "prefix_source": dump_texture_collection(tuple(adopted_drafts)),
        "provenance": provenance,
        "normalization": normalization,
        "projection": projection,
        "prefix_validation": validation,
    }


def prepare_supported_normal_generation_v16(
    project_root: Path,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """外部呼出し前にV15bの来歴と前半伴奏を再検査し、V16を準備する。"""

    project_root = Path(project_root).resolve()
    artifact_root = (
        (project_root / DEFAULT_ARTIFACT_NAME).resolve()
        if artifact_root is None
        else Path(artifact_root).resolve()
    )
    if artifact_root.exists():
        raise SupportedNormalGenerationV16Error(
            f"V16 artifact root already exists: {artifact_root}"
        )
    preflight = _build_preflight(project_root)
    inputs = artifact_root / "inputs"
    inputs.mkdir(parents=True)
    harmony_path = inputs / "prepared-harmonic-collection.dsl"
    melody_path = inputs / "prepared-melody-collection.dsl"
    budget_path = inputs / "prepared-texture-budget.json"
    prefix_path = inputs / "prepared-texture-prefix.dsl"
    provenance_path = inputs / "prepared-texture-prefix-provenance.json"
    harmony_path.write_bytes(
        (preflight["source_run"] / "inputs/prepared-harmonic-collection.dsl").read_bytes()
    )
    melody_path.write_text(
        dump_melody_collection(preflight["normalized_melodies"]),
        encoding="utf-8",
    )
    atomic_write_json(budget_path, preflight["texture_budget"])
    prefix_path.write_text(preflight["prefix_source"], encoding="utf-8")
    atomic_write_json(provenance_path, preflight["provenance"])
    whole_run = artifact_root / "runs/whole-score"
    prepared = prepare_whole_score_live_run(
        project_root,
        whole_run,
        source=WholeScorePreparedSource(
            run_root=project_root / SOURCE_RUN_NAME,
            harmonic_collection_path=harmony_path,
            melody_collection_path=melody_path,
            texture_budget_path=budget_path,
            texture_collection_prefix_path=prefix_path,
            texture_prefix_provenance_path=provenance_path,
            texture_batch_maximum_event_count=BATCH_MAXIMUM_EVENT_COUNT,
            maximum_external_calls=MAXIMUM_EXTERNAL_CALLS,
            register_enforcement_mode="hard",
        ),
    )
    result = {
        "schema_version": 1,
        "status": prepared["status"],
        "external_calls_made": 0,
        "maximum_external_calls": MAXIMUM_EXTERNAL_CALLS,
        "adopted_material_ids": list(ADOPTED_MATERIAL_IDS),
        "excluded_material_ids": list(EXCLUDED_MATERIAL_IDS),
        "melody_ending_normalization": preflight["normalization"],
        "melody_projection": preflight["projection"],
        "prefix_validation_sha256": sha256_json(preflight["prefix_validation"]),
        "prepared_texture_prefix_sha256": sha256_file(prefix_path),
        "prepared_texture_prefix_provenance_sha256": sha256_file(provenance_path),
        "whole_score": prepared,
    }
    atomic_write_json(artifact_root / "manifest.json", result)
    return result


def execute_supported_normal_generation_v16(
    project_root: Path,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """準備済みV16で最終伴奏と演奏を各1回だけ生成する。"""

    project_root = Path(project_root).resolve()
    artifact_root = (
        (project_root / DEFAULT_ARTIFACT_NAME).resolve()
        if artifact_root is None
        else Path(artifact_root).resolve()
    )
    manifest_path = artifact_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "prepared":
        raise SupportedNormalGenerationV16Error("V16 run is not prepared")
    whole_run = artifact_root / "runs/whole-score"
    runner = create_default_whole_score_runner(
        project_root, whole_run, maximum_external_calls=MAXIMUM_EXTERNAL_CALLS
    )
    try:
        result = execute_prepared_whole_score_live_run(project_root, whole_run, runner)
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "external_calls_made": runner.call_number,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "same_run_retry": False,
                },
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise
    manifest.update(
        {
            "status": result.get("status", "failed"),
            "external_calls_made": runner.call_number,
            "whole_score": result,
        }
    )
    status = manifest["status"]
    if status in {"completed_fit", "completed_unfit"}:
        relative = (
            Path("outputs/final.mid") if status == "completed_fit" else Path("staged/final.mid")
        )
        smf = whole_run / relative
        if not smf.is_file():
            raise SupportedNormalGenerationV16Error("completed V16 is missing SMF")
        manifest["smf"] = {"path": str(smf), "sha256": sha256_file(smf)}
    atomic_write_json(manifest_path, manifest)
    return manifest
