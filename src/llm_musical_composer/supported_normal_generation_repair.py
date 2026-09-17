"""検証済み6素材を固定し、最終releaseと演奏だけを再実行する。"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from llm_musical_composer.performance_pipeline import PiecePlan
from llm_musical_composer.run_state import atomic_write_bytes, atomic_write_json, sha256_file
from llm_musical_composer.whole_score_staged_generation import (
    WholeHarmonicMaterialDraftV0,
    assemble_whole_score_melodies,
    assemble_whole_score_skeleton,
    build_whole_score_context,
)
from llm_musical_composer.whole_score_staged_generation_dsl import (
    dump_texture_collection,
    parse_texture_collection,
)
from llm_musical_composer.whole_score_staged_generation_run import (
    WholeScoreLiveRunError,
    WholeScorePreparedSource,
    create_default_whole_score_runner,
    execute_prepared_whole_score_live_run,
    maximum_attack_group_capacity,
    normalize_harmonic_capacity,
    prepare_whole_score_live_run,
    rescale_harmonic_material,
    validate_early_ending,
    validate_melody_ending,
)

PROTOCOL_ID = "supported-normal-generation-v7-final-release-only"
DEFAULT_ARTIFACT_ROOT = Path(
    ".appendix/supported-normal-generation-v7-final-release-only"
)
V1_ROOT = Path(".appendix/supported-normal-generation-v1")
V3_ROOT = Path(".appendix/supported-normal-generation-v3-score-group-capacity")
V4_ROOT = Path(".appendix/supported-normal-generation-v4-texture-batches")
V5_ROOT = Path(".appendix/supported-normal-generation-v5-placement-backtracking")
V6_ROOT = Path(".appendix/supported-normal-generation-v6-ending-slack")
V1_PIECE_PLAN_SHA256 = (
    "75b5a4e013941196ee843eb6f62df806086686614746f426f50b906c9fbaa102"
)
V1_HARMONY_SHA256 = (
    "38fc7cb3ac8c8c3aba37c9e5f84614d6649ca456627c14fb003b9ee7dd0e4b39"
)
MINIMUM_ATTACK_GROUP_COUNT = 649
V3_HARMONY_SHA256 = (
    "799006aa97d0e88c4368d826f3a04fdfb3ec43765327f56aa13fb7efee372dc7"
)
V3_MELODY_SHA256 = (
    "620b241d199e8a7969406964efc63d972ab51c56757c25677c5efcc6919ed5f9"
)
V3_TEXTURE_BUDGET_SHA256 = (
    "671f990a99a4801239c61ae5f5d6a8a68a13d93d3786afbf138c546417711a2c"
)
TEXTURE_BATCH_MAXIMUM_EVENT_COUNT = 250
V4_BATCH_1_SHA256 = (
    "e9689b40a05f9d4793f7654f725faa7262c1ff1bd1d90776e6ecb517e85fa8b1"
)
V4_BATCH_2_SHA256 = (
    "e7c952d2895920ab4645c800ffb4a02277789c4f6322dc7c5d90c1bf846d3dc8"
)
V5_PREFIX_SHA256 = (
    "90c1f40d985fc5f3d7517f0a3c3ec38ee2c0a24a1e5e4119dab3c0fbab4c3924"
)
V5_BATCH_3_RAW_SHA256 = (
    "889a02f44861583365052e0c1a4203ebc8dff79d4ef20bd3a35aa247ca91da53"
)
V5_BATCH_3_CANONICAL_SHA256 = (
    "8b66c87fe0ab64a9780642495d0ebd7379b508d40e96ab0d48c6eaa4984ef424"
)
V6_HARMONY_SHA256 = (
    "941e1d9c8ce8d6f9bb0101feccdaeaf3557cfc48ff8154ead2fedc0b84c0a526"
)
V6_MELODY_SHA256 = (
    "cc94e17a85e228d1cf57f1ef67d4ade092c9badd373f6d5a9c1643249b24fb3a"
)
V6_TEXTURE_BUDGET_SHA256 = (
    "3e82a009a0f9563d6a2f3fbe070875aa568c29cc59f0dbfacc4d6fbacaa79567"
)
V6_REPAIR_DIAGNOSTIC_SHA256 = (
    "4f4de057518dcc430dba14d3e5e0e862040092661d3096e13f15e498f8e7b540"
)
V7_PREFIX_SHA256 = (
    "389e381d6d18269aaca3d752860ea7d5d4cddeb6f1f244e902aa2b6685bf0a15"
)
ENDING_EXTENSION_UNITS = 7
MAXIMUM_EXTERNAL_CALLS = 2


class SupportedNormalGenerationRepairError(ValueError):
    """固定したv1入力または容量修復の契約に違反した。"""


def _assemble(
    plan: PiecePlan,
    drafts: tuple[WholeHarmonicMaterialDraftV0, ...],
):
    return assemble_whole_score_skeleton(
        "supported-normal-generation-v2-capacity",
        plan,
        tuple((item.length_units, item.draft) for item in drafts),
    )


def _rescale_material(
    item: WholeHarmonicMaterialDraftV0,
    new_length: int,
) -> WholeHarmonicMaterialDraftV0:
    try:
        return rescale_harmonic_material(item, new_length)
    except WholeScoreLiveRunError as error:
        raise SupportedNormalGenerationRepairError(str(error)) from error


def repair_harmonic_capacity(
    plan: PiecePlan,
    source: tuple[WholeHarmonicMaterialDraftV0, ...],
    *,
    minimum_attack_group_count: int,
    reserve_ending_hold: bool = True,
) -> tuple[tuple[WholeHarmonicMaterialDraftV0, ...], dict[str, Any]]:
    """和声内容を保ち、容量段の比例拡張だけで厳密予算へ到達させる。"""

    try:
        repaired, normalized = normalize_harmonic_capacity(
            plan,
            source,
            minimum_attack_group_count=minimum_attack_group_count,
            reserve_ending_hold=reserve_ending_hold,
        )
    except WholeScoreLiveRunError as error:
        raise SupportedNormalGenerationRepairError(str(error)) from error
    return repaired, {
        "schema_version": normalized["schema_version"],
        "minimum_attack_group_count": normalized["minimum_attack_group_count"],
        "reserve_ending_hold": normalized["reserve_ending_hold"],
        "allocation_policy": normalized["allocation_policy"],
        "source_capacity": normalized["source_capacity"],
        "source_early_ending": normalized["source_early_ending"],
        "repaired_capacity": normalized["effective_capacity"],
        "repaired_early_ending": normalized["effective_early_ending"],
        "materials": [
            {
                "material_id": item["material_id"],
                "source_length_units": item["source_length_units"],
                "repaired_length_units": item["effective_length_units"],
            }
            for item in normalized["materials"]
        ],
        "content_invariants": normalized["content_invariants"],
    }


def extend_final_release_capacity(
    plan: PiecePlan,
    source: tuple[WholeHarmonicMaterialDraftV0, ...],
    *,
    extension_units: int = ENDING_EXTENSION_UNITS,
) -> tuple[tuple[WholeHarmonicMaterialDraftV0, ...], dict[str, Any]]:
    """単一主和音の最終素材だけを終止保持余白分だけ延長する。"""

    if extension_units <= 0:
        raise SupportedNormalGenerationRepairError(
            "ending extension must be positive"
        )
    context = build_whole_score_context(plan)
    if len(source) != len(context.material_ids) or len(context.release_material_ids) != 1:
        raise SupportedNormalGenerationRepairError(
            "ending extension requires one release material"
        )
    final_material_id = context.release_material_ids[0]
    final_index = context.material_ids.index(final_material_id)
    final = source[final_index]
    if len(final.draft.events) != 1:
        raise SupportedNormalGenerationRepairError(
            "ending extension requires one harmony event"
        )
    extended = list(source)
    extended[final_index] = _rescale_material(
        final,
        final.length_units + extension_units,
    )
    result = tuple(extended)
    skeleton = _assemble(plan, result)
    capacity = maximum_attack_group_capacity(plan, skeleton)
    ending = validate_early_ending(plan, skeleton)
    return result, {
        "schema_version": 1,
        "policy": "extend-dedicated-final-release-v1",
        "material_id": final_material_id,
        "source_length_units": final.length_units,
        "extended_length_units": extended[final_index].length_units,
        "extension_units": extension_units,
        "capacity": capacity,
        "early_ending": ending,
    }


def align_final_melody_to_ending(
    plan: PiecePlan,
    skeleton,
    source,
):
    """既存最終主音だけを新しい最終発音可能位置へ移す。"""

    context = build_whole_score_context(plan)
    if len(source) != len(context.material_ids) or len(context.release_material_ids) != 1:
        raise SupportedNormalGenerationRepairError(
            "ending melody alignment requires one release material"
        )
    final_material_id = context.release_material_ids[0]
    final_index = context.material_ids.index(final_material_id)
    skeleton_material = skeleton.materials[final_index]
    draft = source[final_index]
    if not draft.events:
        raise SupportedNormalGenerationRepairError("ending melody is empty")
    final_attack = max(event.at_units for event in draft.events)
    final_events = [event for event in draft.events if event.at_units == final_attack]
    if len(final_events) != 1:
        raise SupportedNormalGenerationRepairError(
            "ending melody must have one final event"
        )
    final_event = final_events[0]
    total_units = sum(
        skeleton.materials[index].length_units
        * len(context.occurrence_weights[material_id])
        for index, material_id in enumerate(context.material_ids)
    )
    minimum_duration = (total_units * 2_000 + 180_000 - 1) // 180_000
    expected_attack = skeleton_material.length_units - minimum_duration
    if (
        final_event.pitch % 12 != plan.tonal_center
        or final_event.duration_units < minimum_duration
        or final_event.at_units + final_event.duration_units
        != skeleton_material.length_units - ENDING_EXTENSION_UNITS
    ):
        raise SupportedNormalGenerationRepairError(
            "source ending melody is not the fixed valid ending"
        )
    events = tuple(
        replace(event, at_units=expected_attack)
        if event is final_event
        else event
        for event in draft.events
    )
    aligned = list(source)
    aligned[final_index] = replace(draft, events=events)
    result = tuple(aligned)
    payload = assemble_whole_score_melodies(
        "supported-normal-generation-v6-ending",
        plan,
        skeleton,
        result,
    )
    validation = validate_melody_ending(plan, skeleton, payload)
    return result, {
        "schema_version": 1,
        "policy": "move-final-tonic-to-latest-usable-onset-v1",
        "material_id": final_material_id,
        "source_attack_units": final_attack,
        "aligned_attack_units": expected_attack,
        "duration_units": final_event.duration_units,
        "validation": validation,
    }


def _write_immutable(path: Path, value: bytes, label: str) -> None:
    if path.is_file():
        if path.read_bytes() != value:
            raise SupportedNormalGenerationRepairError(f"saved {label} conflicts")
        return
    atomic_write_bytes(path, value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupportedNormalGenerationRepairError(f"JSON object required: {path}")
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def prepare_supported_normal_generation_repair(
    project_root: Path,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """v5とv6の検証済み成果を固定し、最終releaseだけ残すrunを準備する。"""

    project_root = Path(project_root).resolve()
    artifact_root = Path(artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = project_root / artifact_root
    artifact_root = artifact_root.resolve()
    v1_root = project_root / V1_ROOT
    v3_root = project_root / V3_ROOT
    v4_root = project_root / V4_ROOT
    v5_root = project_root / V5_ROOT
    v6_root = project_root / V6_ROOT
    v1_manifest = _read_json(v1_root / "manifest.json")
    if v1_manifest.get("status") != "failed":
        raise SupportedNormalGenerationRepairError("v1 run is not the fixed failure")
    v3_manifest = _read_json(v3_root / "manifest.json")
    if v3_manifest.get("status") != "failed":
        raise SupportedNormalGenerationRepairError("v3 run is not the fixed failure")
    v4_manifest = _read_json(v4_root / "manifest.json")
    if (
        v4_manifest.get("status") != "failed"
        or v4_manifest.get("confirmed_external_call_count") != 2
    ):
        raise SupportedNormalGenerationRepairError("v4 run is not the fixed failure")
    v5_manifest = _read_json(v5_root / "manifest.json")
    v5_run = v5_root / "runs/whole-score"
    v5_state = _read_json(v5_run / "run-state.json")
    v5_batch_3 = _read_json(v5_run / "steps/texture-collection-batch-003.json")
    v5_raw_batch_3_path = v5_run / "responses/texture-collection-batch-003.dsl"
    v5_canonical_batch_3_path = v5_run / "outputs/texture-collection-batch-003.dsl"
    if (
        v5_manifest.get("status") != "failed"
        or v5_manifest.get("confirmed_external_call_count") != 1
        or v5_state.get("status") != "failed"
        or v5_state.get("calls", {}).get("confirmed_external_call_count") != 1
        or v5_batch_3.get("status") != "completed"
        or v5_batch_3.get("outputs", {}).get("external_call_number") != 1
        or v5_batch_3.get("outputs", {}).get("material_ids")
        != [
            "material_return_variation",
            "material_release_preparation",
            "material_final_release",
        ]
        or v5_batch_3.get("outputs", {}).get("required_texture_event_count") != 322
        or v5_batch_3.get("outputs", {}).get("texture_draft_sha256")
        != V5_BATCH_3_CANONICAL_SHA256
        or "performance-collection" in v5_state.get("steps", {})
        or sha256_file(v5_raw_batch_3_path) != V5_BATCH_3_RAW_SHA256
        or sha256_file(v5_canonical_batch_3_path)
        != V5_BATCH_3_CANONICAL_SHA256
    ):
        raise SupportedNormalGenerationRepairError("v5 ending provenance mismatch")
    v6_manifest = _read_json(v6_root / "manifest.json")
    v6_run = v6_root / "runs/whole-score"
    v6_run_manifest = _read_json(v6_run / "manifest.json")
    v6_state = _read_json(v6_run / "run-state.json")
    v6_failure = _read_json(v6_run / "steps/run-failure.json")
    v6_calls = v6_state.get("calls", {})
    if (
        v6_manifest.get("status") != "failed"
        or v6_manifest.get("confirmed_external_call_count") != 1
        or v6_run_manifest.get("status") != "failed"
        or v6_run_manifest.get("confirmed_external_call_count") != 1
        or v6_run_manifest.get("error", {}).get("type")
        != "InterruptedAttemptError"
        or v6_state.get("status") != "interrupted"
        or v6_failure.get("status") != "failed"
        or v6_failure.get("outputs", {}).get("type") != "InterruptedAttemptError"
        or v6_calls.get("confirmed_external_call_count") != 1
        or v6_calls.get("interrupted_external_call_count") != 1
        or v6_calls.get("successful_external_call_count") != 0
        or v6_calls.get("saved_response_count") != 0
        or "performance-collection" in v6_state.get("steps", {})
        or (v6_run / "outputs/performance-spec.dsl").exists()
        or (v6_run / "responses/performance-collection.dsl").exists()
    ):
        raise SupportedNormalGenerationRepairError("v6 timeout provenance mismatch")
    piece_plan_path = v1_root / "runs/piece-plan/outputs/piece-plan.dsl"
    source_harmony_path = v1_root / "runs/whole-score/responses/harmonic-collection.dsl"
    harmony_path = v3_root / "runs/whole-score/outputs/harmonic-collection.dsl"
    melody_path = v3_root / "runs/whole-score/outputs/melody-collection-main.dsl"
    texture_budget_path = v3_root / "runs/whole-score/outputs/texture-budget.json"
    v4_run = v4_root / "runs/whole-score"
    v4_state = _read_json(v4_run / "run-state.json")
    v4_step_1 = _read_json(v4_run / "steps/texture-collection-batch-001.json")
    v4_step_2 = _read_json(v4_run / "steps/texture-collection-batch-002.json")
    v4_failure_2 = _read_json(
        v4_run / "failures/texture-collection-batch-002.json"
    )
    expected_step_inputs = {
        "harmonic_skeleton": (
            "e9d5b79c105be259d637a4ab2a9413f554a5d3b7c61f2df16fdfa04dc6c23217"
        ),
        "melodies": (
            "02cd9dec23faeceee41cff9b6df754f998cae8b70aeab564eaf1d3cbedf7278f"
        ),
        "prompt_target": (
            "a829774a73d7332a0bed7641b1d722702911a15956cc82379654f6b1b7a048cd"
        ),
        "texture_budget": V3_TEXTURE_BUDGET_SHA256,
    }
    for step in (v4_step_1, v4_step_2):
        if any(
            step.get("input_hashes", {}).get(name) != value
            for name, value in expected_step_inputs.items()
        ):
            raise SupportedNormalGenerationRepairError("v4 step input hash mismatch")
    if (
        v4_state.get("status") != "failed"
        or v4_state.get("calls", {}).get("confirmed_external_call_count") != 2
        or v4_step_1.get("status") != "completed"
        or v4_step_2.get("status") != "failed"
        or v4_step_1.get("input_hashes", {}).get("batch_material_ids")
        != "5dfc69a79452dcc2d542a78e0da2c32d2e36d470e4d5e87d8dd9681319827e49"
        or v4_step_2.get("input_hashes", {}).get("batch_material_ids")
        != "b2be724436927efdb3e85cfaff0454a7c054a0ed68ac4f5d2f30edf6bc098a20"
        or v4_step_1.get("outputs", {}).get("texture_draft_sha256")
        != V4_BATCH_1_SHA256
        or v4_failure_2.get("source_sha256") != V4_BATCH_2_SHA256
        or (v4_run / "attempts/texture-collection-batch-003").exists()
    ):
        raise SupportedNormalGenerationRepairError("v4 batch provenance mismatch")
    if sha256_file(piece_plan_path) != V1_PIECE_PLAN_SHA256:
        raise SupportedNormalGenerationRepairError("v1 PiecePlan hash mismatch")
    if sha256_file(source_harmony_path) != V1_HARMONY_SHA256:
        raise SupportedNormalGenerationRepairError("v1 harmony hash mismatch")
    expected_hashes = {
        harmony_path: V3_HARMONY_SHA256,
        melody_path: V3_MELODY_SHA256,
        texture_budget_path: V3_TEXTURE_BUDGET_SHA256,
    }
    for path, expected_hash in expected_hashes.items():
        if sha256_file(path) != expected_hash:
            raise SupportedNormalGenerationRepairError(
                f"fixed v3 input hash mismatch: {path.name}"
            )
    repaired_path = artifact_root / "inputs/repaired-harmonic-collection.dsl"
    prepared_melody_path = artifact_root / "inputs/prepared-melody-collection.dsl"
    prepared_texture_budget_path = artifact_root / "inputs/prepared-texture-budget.json"
    prepared_texture_prefix_path = artifact_root / "inputs/prepared-texture-prefix.dsl"
    legacy_prefix_provenance_path = (
        artifact_root / "inputs/legacy-prefix-provenance.json"
    )
    diagnostic_path = artifact_root / "inputs/ending-capacity-repair.json"
    v6_harmony_path = v6_root / "inputs/repaired-harmonic-collection.dsl"
    v6_melody_path = v6_root / "inputs/prepared-melody-collection.dsl"
    v6_budget_path = v6_root / "inputs/prepared-texture-budget.json"
    v6_diagnostic_path = v6_root / "inputs/ending-capacity-repair.json"
    v6_fixed_hashes = {
        v6_harmony_path: V6_HARMONY_SHA256,
        v6_melody_path: V6_MELODY_SHA256,
        v6_budget_path: V6_TEXTURE_BUDGET_SHA256,
        v6_diagnostic_path: V6_REPAIR_DIAGNOSTIC_SHA256,
    }
    for path, expected_hash in v6_fixed_hashes.items():
        if sha256_file(path) != expected_hash:
            raise SupportedNormalGenerationRepairError(
                f"fixed v6 input hash mismatch: {path.name}"
            )
    _write_immutable(
        repaired_path,
        v6_harmony_path.read_bytes(),
        "repaired harmony",
    )
    _write_immutable(
        prepared_melody_path,
        v6_melody_path.read_bytes(),
        "prepared melody",
    )
    _write_immutable(
        prepared_texture_budget_path,
        v6_budget_path.read_bytes(),
        "prepared texture budget",
    )
    _write_immutable(
        diagnostic_path,
        v6_diagnostic_path.read_bytes(),
        "ending repair diagnostic",
    )
    batch_1_path = v4_run / "outputs/texture-collection-batch-001.dsl"
    batch_2_path = v4_run / "responses/texture-collection-batch-002.dsl"
    if sha256_file(batch_1_path) != V4_BATCH_1_SHA256:
        raise SupportedNormalGenerationRepairError("v4 batch 1 hash mismatch")
    if sha256_file(batch_2_path) != V4_BATCH_2_SHA256:
        raise SupportedNormalGenerationRepairError("v4 batch 2 hash mismatch")
    prefix_source = dump_texture_collection(
        (
            *parse_texture_collection(batch_1_path.read_text(encoding="utf-8")),
            *parse_texture_collection(batch_2_path.read_text(encoding="utf-8")),
            *parse_texture_collection(
                v5_canonical_batch_3_path.read_text(encoding="utf-8")
            )[:2],
        )
    )
    _write_immutable(
        prepared_texture_prefix_path,
        prefix_source.encode("utf-8"),
        "prepared texture prefix",
    )
    if sha256_file(prepared_texture_prefix_path) != V7_PREFIX_SHA256:
        raise SupportedNormalGenerationRepairError("v7 texture prefix hash mismatch")
    source_target_path = v1_root / "runs/piece-plan/inputs/prompt-target.json"
    source_spec_path = v1_root / "runs/piece-plan/run-spec.json"
    source_target = _read_json(source_target_path)
    legacy_prefix_provenance = {
        "schema_version": 1,
        "target_version": source_target["target_version"],
        "prepared_texture_prefix_sha256": sha256_file(
            prepared_texture_prefix_path
        ),
        "source_prompt_target": {
            "path": source_target_path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(source_target_path),
        },
        "source_run_spec": {
            "path": source_spec_path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(source_spec_path),
        },
    }
    _write_immutable(
        legacy_prefix_provenance_path,
        _json_bytes(legacy_prefix_provenance),
        "legacy prefix provenance",
    )
    whole_run = artifact_root / "runs/whole-score"
    prepared = prepare_whole_score_live_run(
        project_root,
        whole_run,
        source=WholeScorePreparedSource(
            run_root=v1_root / "runs/piece-plan",
            harmonic_collection_path=repaired_path,
            melody_collection_path=prepared_melody_path,
            texture_budget_path=prepared_texture_budget_path,
            texture_collection_prefix_path=prepared_texture_prefix_path,
            texture_batch_maximum_event_count=(
                TEXTURE_BATCH_MAXIMUM_EVENT_COUNT
            ),
            maximum_external_calls=MAXIMUM_EXTERNAL_CALLS,
            register_enforcement_mode="legacy_prefix_diagnostic_only",
            legacy_prefix_provenance_path=legacy_prefix_provenance_path,
        ),
    )
    manifest_path = artifact_root / "manifest.json"
    if manifest_path.is_file():
        return _read_json(manifest_path)
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "prepared",
        "passes": None,
        "promoted": False,
        "maximum_external_call_count": MAXIMUM_EXTERNAL_CALLS,
        "piece_plan_external_call_count": 0,
        "harmony_external_call_count": 0,
        "melody_external_call_count": 0,
        "texture_batch_maximum_event_count": (
            TEXTURE_BATCH_MAXIMUM_EVENT_COUNT
        ),
        "whole_score_preflight": prepared,
        "repair_diagnostic_sha256": sha256_file(diagnostic_path),
        "repaired_harmonic_collection_sha256": sha256_file(repaired_path),
        "prepared_melody_collection_sha256": sha256_file(prepared_melody_path),
        "prepared_texture_budget_sha256": sha256_file(
            prepared_texture_budget_path
        ),
        "prepared_texture_prefix_sha256": sha256_file(
            prepared_texture_prefix_path
        ),
        "source_texture_batch_sha256": [
            V4_BATCH_1_SHA256,
            V4_BATCH_2_SHA256,
        ],
        "source_v5_texture_batch_sha256": {
            "raw_response": V5_BATCH_3_RAW_SHA256,
            "canonical_output": V5_BATCH_3_CANONICAL_SHA256,
            "reused_material_count": 2,
        },
        "source_v6_input_sha256": {
            "harmony": V6_HARMONY_SHA256,
            "melody": V6_MELODY_SHA256,
            "texture_budget": V6_TEXTURE_BUDGET_SHA256,
            "repair_diagnostic": V6_REPAIR_DIAGNOSTIC_SHA256,
        },
        "repair_implementation_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def run_supported_normal_generation_repair(
    project_root: Path,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """固定6素材から最終release伴奏と演奏を再試行なしで実行する。"""

    project_root = Path(project_root).resolve()
    artifact_root = Path(artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = project_root / artifact_root
    artifact_root = artifact_root.resolve()
    manifest = prepare_supported_normal_generation_repair(project_root, artifact_root)
    if manifest.get("status") != "prepared":
        return manifest
    run_dir = artifact_root / "runs/whole-score"
    runner = create_default_whole_score_runner(
        project_root,
        run_dir,
        maximum_external_calls=MAXIMUM_EXTERNAL_CALLS,
    )
    result = execute_prepared_whole_score_live_run(project_root, run_dir, runner)
    manifest.update(
        {
            "status": result["status"],
            "passes": bool(result.get("passes")),
            "promoted": bool(result.get("promoted")),
            "confirmed_external_call_count": runner.call_number,
            "whole_score_result": result,
        }
    )
    staged_smf = run_dir / "staged/final.mid"
    if staged_smf.is_file():
        manifest["staged_smf_path"] = str(staged_smf.relative_to(artifact_root))
        manifest["staged_smf_sha256"] = sha256_file(staged_smf)
    atomic_write_json(artifact_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        result = prepare_supported_normal_generation_repair(
            arguments.project_root, arguments.artifact_root
        )
        success = result.get("status") == "prepared"
    else:
        result = run_supported_normal_generation_repair(
            arguments.project_root, arguments.artifact_root
        )
        success = result.get("status") == "completed_fit"
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
