"""曲全体の段階生成を外部実走する前後の検査を扱う。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

from llm_musical_composer.control_reference_baseline import (
    extract_control_observables,
    normalize_value,
)
from llm_musical_composer.generation_intent import (
    candidate_use_allows_promotion,
    creative_targets_for_step,
    merge_generation_intent,
)
from llm_musical_composer.generic_pipeline_quality import (
    LOW_PITCH_BOUNDARY,
    MINIMUM_LOW_SPACING_SEMITONES,
    evaluate_generic_piece_plan_quality,
    evaluate_generic_pipeline_quality,
    evaluate_generic_score_quality,
)
from llm_musical_composer.harmonic_skeleton import (
    HarmonicSkeletonV0,
    ScorePayloadV0,
    apply_harmonic_skeleton,
    dump_harmonic_skeleton,
    validate_harmonic_skeleton,
)
from llm_musical_composer.performance_pipeline import (
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    RenderedPerformance,
    ScoreSpec,
    analyze_voice_velocity,
    dynamic_velocity_unclamped,
    material_velocity_adjustments,
    render_musicxml,
    render_performance,
    render_performance_smf,
    resolve_effective_profile,
    validate_musicxml_round_trip,
    validate_score_spec,
    validate_smf_round_trip,
)
from llm_musical_composer.piano_texture_register_placement import (
    REGISTER_ZONES,
    PianoTexturePlacementResultV6,
    PianoTexturePlacementResultV7,
    PianoTexturePlacementResultV8,
)
from llm_musical_composer.pilot_loop import (
    CodexExecRunner,
    isolated_codex_working_directory,
)
from llm_musical_composer.pipeline_dsl import (
    dump_performance_spec,
    dump_score_spec,
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.recurrence_quality import (
    analyze_foreground_dissonance,
    analyze_material_harmony,
    analyze_material_vertical_alignment,
)
from llm_musical_composer.reference_generation_target import (
    LEGACY_TARGET_VERSION,
    build_reference_generation_target,
)
from llm_musical_composer.reference_profile import (
    build_copy_fingerprint,
    evaluate_copy_risk,
    extract_reference_profile,
    load_reference_piece,
)
from llm_musical_composer.run_state import (
    RunStore,
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    sha256_json,
    sha256_text,
)
from llm_musical_composer.staged_material_pilot import (
    HarmonicEventDraft,
    assemble_texture,
    build_texture_feasibility,
    full_low_spacing_violations,
    texture_budget_onset_capacity_reachability,
    texture_preplacement_violations,
)
from llm_musical_composer.texture_budget import (
    allocate_texture_budget,
    material_attack_group_capacities,
    measure_rendered_texture_budget,
    measure_score_texture_budget,
)
from llm_musical_composer.tonal_hierarchy import (
    evaluate_brightness_resolution,
    evaluate_harmonic_skeleton_tonal_hierarchy,
)
from llm_musical_composer.whole_score_staged_generation import (
    WHOLE_SCORE_TARGET_DURATION_MS,
    WholeHarmonicMaterialDraftV0,
    WholeScoreTextureResultV0,
    analyze_foreground_transition,
    assemble_whole_score_melodies,
    assemble_whole_score_performance,
    assemble_whole_score_skeleton,
    assemble_whole_score_textures,
    build_whole_score_context,
    melody_generation_order,
)
from llm_musical_composer.whole_score_staged_generation_dsl import (
    dump_harmonic_collection,
    dump_melody_collection,
    dump_performance_collection,
    dump_texture_collection,
    parse_harmonic_collection,
    parse_melody_collection,
    parse_performance_collection,
    parse_texture_collection,
)


class WholeScoreLiveRunError(ValueError):
    """曲全体の外部実走契約に違反した入力。"""


class _TextureBatchPlacementError(WholeScoreLiveRunError):
    """伴奏batchの配置不能を構造化して伝える。"""

    def __init__(self, material_id: str, placement: Any) -> None:
        self.material_id = material_id
        self.placement_schema_version = (
            4
            if isinstance(placement, PianoTexturePlacementResultV8)
            else 3
            if isinstance(placement, PianoTexturePlacementResultV7)
            else 2
            if isinstance(placement, PianoTexturePlacementResultV6)
            else 1
        )
        self.placement = json.loads(
            json.dumps(asdict(placement), ensure_ascii=False)
        )
        super().__init__(
            "texture batch material cannot be placed: "
            f"{material_id}: {self.placement['status']}"
        )


PROTOCOL_ID = "whole-score-staged-generation-live-v6"
MODEL_CONFIG = {
    "model": "gpt-5.6-sol",
    "reasoning_effort": "high",
    "timeout_seconds": 900,
    "maximum_external_calls": 5,
}
DEFAULT_MAXIMUM_EXTERNAL_CALLS = 5
MAXIMUM_EXTERNAL_CALLS = 9
ATTACK_FREQUENCY_TARGET = 2.72991994
ATTACK_FREQUENCY_P25 = 1.76465137
ATTACK_FREQUENCY_P75 = 3.80406164
V7_ACCEPTED_ATTACK_GROUP_COUNT = 314
REGISTER_LOWER_BOUND_EXPERIMENT_POLICY_ID = (
    "register-lower-bound-experiment-v1"
)
_REGISTER_EXCEPTION_POLICIES: dict[str, dict[str, Any]] = {
    REGISTER_LOWER_BOUND_EXPERIMENT_POLICY_ID: {
        "input_sha256": {
            "piece_plan": (
                "425b36623c78453cd462f5b9ef4795ef32e45745bf6ef44ba88af58d3db2e77d"
            ),
            "prompt_target": (
                "8aae2de74a3322f788ef7fa2dfc883fa36ce3d855c748226fff399b062409144"
            ),
            "harmonic_collection": (
                "781dfa6ebc7834c012d4ff0fe84785f867dc52c03f25b9b6d5eed97b0ba345a5"
            ),
            "melody_collection": (
                "6c8b5d0987d7966e2beb45a0837ed46e7b64fe23d6c7c3b041b2a7ce5a2469fb"
            ),
            "texture_budget": (
                "8906b61a03b6ad6a056364a1cfaf3e37c07f3d434be447f8c9f3e36b03d16341"
            ),
            "source_failure": (
                "7876806a809c03c6020e3e791ea1c0f152745358f4eb534f84537ff4eb82788a"
            ),
        },
        "texture_batch_maximum_event_count": 400,
        "normal_allowed_pitch_range": [48, 96],
        "experimental_allowed_pitch_range": [21, 96],
        "entry_material_id": "material_1",
        "full_batch_failure": {
            "material_id": "material_2",
            "placement_status": "search_unplaceable",
        },
    }
}

_SOURCE_RUN = Path(
    ".appendix/reference-variance-smoke-v7/runs/reference-a-candidate-2"
)
_PROMPTS = (
    Path("prompts/pipeline-whole-harmonic-collection.md"),
    Path("prompts/pipeline-whole-melody-collection.md"),
    Path("prompts/pipeline-whole-texture-collection.md"),
    Path("prompts/pipeline-whole-performance-collection.md"),
)
_IMPLEMENTATIONS = (
    Path("src/llm_musical_composer/whole_score_staged_generation.py"),
    Path("src/llm_musical_composer/whole_score_staged_generation_dsl.py"),
    Path("src/llm_musical_composer/whole_score_staged_generation_run.py"),
    Path("src/llm_musical_composer/pipeline_dsl.py"),
    Path("src/llm_musical_composer/harmonic_skeleton.py"),
    Path("src/llm_musical_composer/staged_material_pilot.py"),
    Path("src/llm_musical_composer/piano_texture_register_placement.py"),
    Path("src/llm_musical_composer/piano_texture_pilot.py"),
    Path("src/llm_musical_composer/performance_pipeline.py"),
    Path("src/llm_musical_composer/generic_pipeline_quality.py"),
    Path("src/llm_musical_composer/recurrence_analysis.py"),
    Path("src/llm_musical_composer/recurrence_quality.py"),
    Path("src/llm_musical_composer/reference_profile.py"),
    Path("src/llm_musical_composer/reference_generation_target.py"),
    Path("src/llm_musical_composer/generation_intent.py"),
    Path("src/llm_musical_composer/tonal_hierarchy.py"),
    Path("src/llm_musical_composer/texture_budget.py"),
    Path("src/llm_musical_composer/smf_notes.py"),
    Path("src/llm_musical_composer/control_axis_anchors.py"),
    Path("src/llm_musical_composer/control_reference_baseline.py"),
    Path("src/llm_musical_composer/pilot_loop.py"),
    Path("src/llm_musical_composer/run_state.py"),
    Path("schemas/codex-composition-response.schema.json"),
    *_PROMPTS,
)


class WholeScoreStageRunner(Protocol):
    @property
    def call_number(self) -> int: ...

    def run(
        self,
        step_id: str,
        prompt: str,
        input_hashes: dict[str, str] | None = None,
    ) -> dict[str, object]: ...


@dataclass(frozen=True)
class LivePerformanceStageResult:
    performance: PerformanceSpec
    rendered: RenderedPerformance
    score_quality: dict[str, Any]
    pipeline_quality: dict[str, Any]
    musicxml_path: Path
    smf_path: Path


@dataclass(frozen=True)
class WholeScorePreparedSource:
    """曲全体生成へ渡す、PiecePlan段階まで確定したrun。"""

    run_root: Path
    use_legacy_v7_frequency: bool = False
    source_smf_relative: Path | None = None
    harmonic_collection_path: Path | None = None
    melody_collection_path: Path | None = None
    texture_budget_path: Path | None = None
    texture_collection_prefix_path: Path | None = None
    texture_batch_maximum_event_count: int | None = None
    texture_batch_maximum_count: int | None = None
    maximum_external_calls: int | None = None
    generation_profile_id: str = "legacy-v1"
    velocity_policy_id: str = "legacy-unison-v1"
    key_release_unreachable_policy: str = "raise"
    register_enforcement_mode: str = "hard"
    texture_prefix_provenance_path: Path | None = None
    legacy_prefix_provenance_path: Path | None = None
    register_exception_provenance_path: Path | None = None


def _target_version_number(target_version: str) -> int:
    prefix = "reference-generation-target-v"
    if not target_version.startswith(prefix):
        raise WholeScoreLiveRunError("register target version is invalid")
    try:
        value = int(target_version.removeprefix(prefix))
    except ValueError as error:
        raise WholeScoreLiveRunError("register target version is invalid") from error
    if value < 1:
        raise WholeScoreLiveRunError("register target version is invalid")
    return value


def resolve_register_enforcement(
    *,
    requested_mode: str,
    target_version: str,
    has_prepared_prefix: bool,
    has_verified_legacy_provenance: bool,
) -> dict[str, Any]:
    """明示要求と来歴からrunの音域強制方式を一意に決める。"""

    allowed = {
        "hard",
        "legacy_prefix_diagnostic_only",
        "diagnostic_only_lower_bound_experiment",
    }
    if requested_mode not in allowed:
        raise WholeScoreLiveRunError("register enforcement mode is invalid")
    target_number = _target_version_number(target_version)
    if requested_mode == "legacy_prefix_diagnostic_only" and (
        not has_prepared_prefix
        or not has_verified_legacy_provenance
        or target_number >= 4
    ):
        raise WholeScoreLiveRunError("legacy prefix provenance is invalid")
    if (
        requested_mode == "diagnostic_only_lower_bound_experiment"
        and has_prepared_prefix
    ):
        raise WholeScoreLiveRunError(
            "register lower-bound experiment cannot use a prepared prefix"
        )
    return {
        "requested_mode": requested_mode,
        "effective_mode": requested_mode,
        "promotion_blocker": (
            requested_mode == "diagnostic_only_lower_bound_experiment"
        ),
    }


def apply_register_promotion_policy(
    base_quality_passes: bool,
    register_enforcement: Mapping[str, Any],
) -> dict[str, Any]:
    """通常品質と実験上の昇格阻止を分けて最終状態を返す。"""

    blocked = bool(register_enforcement.get("promotion_blocker"))
    passes = bool(base_quality_passes) and not blocked
    return {
        "base_quality_passes": bool(base_quality_passes),
        "passes": passes,
        "promoted": passes,
        "status": "completed_fit" if passes else "completed_unfit",
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WholeScoreLiveRunError(f"cannot read JSON input: {path}") from error
    if not isinstance(value, dict):
        raise WholeScoreLiveRunError(f"JSON input must be an object: {path}")
    return value


def _run_maximum_external_calls(run_dir: Path) -> int:
    spec = _read_json(Path(run_dir) / "run-spec.json")
    try:
        value = int(spec["model_config"]["maximum_external_calls"])
    except (KeyError, TypeError, ValueError) as error:
        raise WholeScoreLiveRunError("run external call ceiling is invalid") from error
    if not 0 <= value <= MAXIMUM_EXTERNAL_CALLS:
        raise WholeScoreLiveRunError("run external call ceiling is invalid")
    return value


def _run_store(run_dir: Path) -> RunStore:
    return RunStore(
        run_dir,
        max_calls=_run_maximum_external_calls(run_dir),
    )


def _file_record(project_root: Path, relative_path: Path) -> dict[str, str]:
    path = relative_path if relative_path.is_absolute() else project_root / relative_path
    if not path.is_file():
        raise WholeScoreLiveRunError(f"required input is missing: {relative_path}")
    recorded_path = path if relative_path.is_absolute() else relative_path
    return {"path": recorded_path.as_posix(), "sha256": sha256_file(path)}


def _semantic_attack_frequency(prompt_target: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for items in prompt_target.get("semantic_targets", {}).values()
        for item in items
        if isinstance(item, dict) and item.get("id") == "attack_frequency"
    ]
    if len(matches) != 1:
        raise WholeScoreLiveRunError("attack frequency semantic target is missing")
    item = matches[0]
    try:
        target = float(item["anchor"]["raw_groups_per_second"])
        minimum = float(item["promotion_range"]["minimum_groups_per_second"])
        maximum = float(item["promotion_range"]["maximum_groups_per_second"])
        strict_groups = int(item["strict_score_budget"]["groups"])
        tolerance = int(item["grouping"]["tolerance_ms"])
    except (KeyError, TypeError, ValueError) as error:
        raise WholeScoreLiveRunError("attack frequency semantic target is invalid") from error
    if not 0 < minimum <= target <= maximum or strict_groups <= 0 or tolerance != 30:
        raise WholeScoreLiveRunError("attack frequency semantic target is inconsistent")
    return {
        "target": target,
        "candidate_minimum": minimum,
        "candidate_maximum": maximum,
        "minimum_attack_group_count": strict_groups,
        "group_tolerance_ms": tolerance,
        "source": (
            "explicit_corpus_min_max_v1"
            if isinstance(item.get("resolution"), dict)
            and item["resolution"].get("source") == "explicit_corpus_min_max_v1"
            else "reference_neighborhood_v3"
        ),
    }


def _legacy_v7_attack_frequency() -> dict[str, Any]:
    return {
        "target": ATTACK_FREQUENCY_TARGET,
        "candidate_minimum": ATTACK_FREQUENCY_P25,
        "candidate_maximum": ATTACK_FREQUENCY_P75,
        "minimum_attack_group_count": V7_ACCEPTED_ATTACK_GROUP_COUNT,
        "group_tolerance_ms": 30,
        "source": "legacy_v7_fixed",
    }


def _run_attack_frequency(run_dir: Path) -> dict[str, Any]:
    spec = _read_json(Path(run_dir) / "run-spec.json")
    value = spec.get("attack_frequency")
    if not isinstance(value, dict):
        raise WholeScoreLiveRunError("prepared attack frequency contract is missing")
    return value


def _rendered_texture_budget_passes(
    frequency_contract: Mapping[str, Any],
    rendered_measurement: Mapping[str, Any],
) -> bool:
    """明示制御では群数差を候補評価へ送り、形の破綻だけを段階失敗にする。"""

    if frequency_contract.get("source") == "explicit_corpus_min_max_v1":
        return bool(rendered_measurement.get("texture_shape_matches_budget"))
    return bool(rendered_measurement.get("matches_budget"))


def _run_texture_placement_policy(run_dir: Path) -> str:
    spec = _read_json(Path(run_dir) / "run-spec.json")
    value = spec.get("texture_placement_policy", "bounded-backtracking-v4")
    if value not in {
        "bounded-backtracking-v4",
        "low-foreground-outward-v5",
        "bidirectional-low-spacing-v6",
        "onset-feasible-zone-v7",
        "search-aware-onset-zone-v8",
    }:
        raise WholeScoreLiveRunError("run texture placement policy is invalid")
    return value


def _descriptor_ids(prompt_target: dict[str, Any]) -> tuple[str, ...]:
    try:
        stages = prompt_target["stage_targets"]
        values = tuple(
            str(item["id"])
            for stage in stages.values()
            for item in stage
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise WholeScoreLiveRunError("prompt target descriptors are invalid") from error
    if not values or len(values) != len(set(values)):
        raise WholeScoreLiveRunError("prompt target descriptor IDs are empty or duplicated")
    return values


def _legacy_prompt_target(prompt_target: dict[str, Any]) -> dict[str, Any]:
    """v2の追加領域を除き、保存済みv1との互換部分を返す。"""
    legacy = dict(prompt_target)
    legacy.pop("semantic_targets", None)
    legacy["target_version"] = LEGACY_TARGET_VERSION
    return legacy


def _prompt_target_for_saved_version(
    prompt_target: dict[str, Any],
    saved_version: Any,
    *,
    legacy_attack_frequency_target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """現在版から追加項目だけを除き、保存済み版と比較できる形へ戻す。"""

    if saved_version in {
        "reference-generation-target-v2",
        "reference-generation-target-v3",
        "reference-generation-target-v4",
        "reference-generation-target-v5",
        "reference-generation-target-v6",
    } and legacy_attack_frequency_target is not None:
        compatible = json.loads(json.dumps(prompt_target))
        legacy_matches = [
            item
            for items in legacy_attack_frequency_target.get(
                "semantic_targets", {}
            ).values()
            for item in items
            if item.get("id") == "attack_frequency"
        ]
        if len(legacy_matches) != 1:
            raise WholeScoreLiveRunError(
                "legacy attack frequency semantic target is invalid"
            )
        for items in compatible.get("semantic_targets", {}).values():
            for index, item in enumerate(items):
                if item.get("id") == "attack_frequency":
                    items[index] = legacy_matches[0]
        prompt_target = compatible

    if saved_version == LEGACY_TARGET_VERSION:
        return _legacy_prompt_target(prompt_target)
    if saved_version in {
        "reference-generation-target-v2",
        "reference-generation-target-v3",
    }:
        compatible = json.loads(json.dumps(prompt_target))
        stages = compatible.get("semantic_targets")
        if not isinstance(stages, dict):
            raise WholeScoreLiveRunError("semantic targets are invalid")
        filtered = {
            stage: [
                item
                for item in items
                if item.get("id")
                not in {"tonal_hierarchy", "register_envelope", "velocity_shape"}
            ]
            for stage, items in stages.items()
        }
        compatible["semantic_targets"] = {
            stage: items for stage, items in filtered.items() if items
        }
        compatible["target_version"] = saved_version
        return compatible
    if saved_version == "reference-generation-target-v4":
        compatible = json.loads(json.dumps(prompt_target))
        stages = compatible.get("semantic_targets", {})
        compatible["semantic_targets"] = {
            stage: [item for item in items if item.get("id") != "tonal_hierarchy"]
            for stage, items in stages.items()
            if any(item.get("id") != "tonal_hierarchy" for item in items)
        }
        for items in compatible["semantic_targets"].values():
            for item in items:
                if item.get("id") == "register_envelope":
                    item.pop("policy_id", None)
        compatible["target_version"] = saved_version
        return compatible
    if saved_version == "reference-generation-target-v5":
        compatible = json.loads(json.dumps(prompt_target))
        stages = compatible.get("semantic_targets", {})
        compatible["semantic_targets"] = {
            stage: [item for item in items if item.get("id") != "tonal_hierarchy"]
            for stage, items in stages.items()
            if any(item.get("id") != "tonal_hierarchy" for item in items)
        }
        compatible["target_version"] = saved_version
        return compatible
    if saved_version == "reference-generation-target-v6":
        compatible = json.loads(json.dumps(prompt_target))
        compatible["target_version"] = saved_version
        return compatible
    if saved_version == prompt_target.get("target_version"):
        return prompt_target
    raise WholeScoreLiveRunError("saved prompt target version is unsupported")


def _semantic_target_view(
    prompt_target: dict[str, Any],
    *target_ids: str,
) -> list[dict[str, Any]]:
    stages = prompt_target.get("semantic_targets")
    if stages is None:
        return []
    if not isinstance(stages, dict):
        raise WholeScoreLiveRunError("semantic targets are invalid")
    by_id: dict[str, dict[str, Any]] = {}
    for items in stages.values():
        if not isinstance(items, list):
            raise WholeScoreLiveRunError("semantic target stage is invalid")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise WholeScoreLiveRunError("semantic target is invalid")
            if item["id"] in by_id:
                raise WholeScoreLiveRunError("semantic target IDs are duplicated")
            by_id[item["id"]] = item
    missing = [target_id for target_id in target_ids if target_id not in by_id]
    if missing:
        raise WholeScoreLiveRunError(f"semantic target is missing: {missing[0]}")
    selected = [by_id[target_id] for target_id in target_ids]
    register = by_id.get("register_envelope")
    if register is not None:
        policy_id = register.get("policy_id")
        version = _target_version_number(str(prompt_target.get("target_version", "")))
        if version >= 5 and policy_id is None:
            raise WholeScoreLiveRunError("register envelope policy is missing")
        if policy_id not in {
            None,
            "main-melody-centered-reference-span-v1",
            "melody-containing-reference-span-v2",
        }:
            raise WholeScoreLiveRunError("register envelope policy is unsupported")
    return selected


def _specified_tonal_targets(prompt_target: dict[str, Any]) -> list[dict[str, Any]]:
    stages = prompt_target.get("semantic_targets")
    if not isinstance(stages, dict):
        return []
    found = [
        item
        for items in stages.values()
        if isinstance(items, list)
        for item in items
        if isinstance(item, dict) and item.get("id") == "tonal_hierarchy"
    ]
    if len(found) > 1:
        raise WholeScoreLiveRunError("semantic target IDs are duplicated")
    return found if found and found[0].get("status") == "specified" else []


def _melody_semantic_targets(prompt_target: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *_specified_tonal_targets(prompt_target),
        *_semantic_target_view(prompt_target, "register_envelope"),
    ]


def _expanded_pitch_values(
    plan: PiecePlan,
    pitches_by_material: Mapping[str, Sequence[int]],
) -> list[int]:
    values: list[int] = []
    for leaf in _ordered_leaves(plan):
        material_id = leaf.score_material_id
        if material_id is not None and material_id in pitches_by_material:
            values.extend(pitches_by_material[material_id])
    return values


def measure_register_progress(
    plan: PiecePlan,
    pitches_by_material: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """確定済み素材をPiecePlanの全出現へ展開して音域進捗を返す。"""

    pitches = _expanded_pitch_values(plan, pitches_by_material)
    if not pitches:
        raise WholeScoreLiveRunError("register progress has no confirmed pitches")
    median_pitch = float(statistics.median(pitches))
    minimum = min(pitches)
    maximum = max(pitches)
    return {
        "note_count": len(pitches),
        "median_pitch": median_pitch,
        "minimum_pitch": minimum,
        "maximum_pitch": maximum,
        "pitch_span_semitones": maximum - minimum,
        "minimum_pitch_count": pitches.count(minimum),
        "maximum_pitch_count": pitches.count(maximum),
    }


def build_allowed_pitch_range(
    plan: PiecePlan,
    pitches_by_material: Mapping[str, Sequence[int]],
    target: dict[str, Any],
) -> dict[str, Any]:
    """main旋律の中央音高から後続生成で共用する不変音域を構築する。"""

    try:
        raw_target_span = target["target_span_semitones"]
        raw_maximum_span = target["maximum_span_semitones"]
    except KeyError as error:
        raise WholeScoreLiveRunError("register envelope target is invalid") from error
    if (
        isinstance(raw_target_span, bool)
        or isinstance(raw_maximum_span, bool)
        or not isinstance(raw_target_span, int)
        or not isinstance(raw_maximum_span, int)
    ):
        raise WholeScoreLiveRunError("register envelope target is invalid")
    target_span = raw_target_span
    maximum_span = raw_maximum_span
    if (
        target_span < 0
        or maximum_span < target_span
        or maximum_span > 87
    ):
        raise WholeScoreLiveRunError("register envelope target is invalid")
    progress = measure_register_progress(plan, pitches_by_material)
    preferred_lower = math.floor(
        progress["median_pitch"] - maximum_span / 2 + 0.5
    )
    preferred_lower = min(max(preferred_lower, 21), 108 - maximum_span)
    policy_id = target.get("policy_id", "main-melody-centered-reference-span-v1")
    if policy_id == "main-melody-centered-reference-span-v1":
        lower = preferred_lower
        upper = lower + maximum_span
        reachable = (
            progress["pitch_span_semitones"] <= maximum_span
            and lower <= progress["minimum_pitch"]
            and progress["maximum_pitch"] <= upper
        )
        return {
            "schema_version": 1,
            "policy_id": policy_id,
            "minimum_pitch": lower,
            "maximum_pitch": upper,
            "target_span_semitones": target_span,
            "maximum_span_semitones": maximum_span,
            "reachable_after_melody": reachable,
            "main_melody_progress": progress,
        }
    if policy_id != "melody-containing-reference-span-v2":
        raise WholeScoreLiveRunError("register envelope policy is unsupported")
    containing_lower_minimum = progress["maximum_pitch"] - maximum_span
    containing_lower_maximum = progress["minimum_pitch"]
    feasible_lower_minimum = max(21, containing_lower_minimum)
    feasible_lower_maximum = min(108 - maximum_span, containing_lower_maximum)
    if (
        progress["pitch_span_semitones"] > maximum_span
        or feasible_lower_minimum > feasible_lower_maximum
    ):
        raise WholeScoreLiveRunError("main melody register envelope is unreachable")
    lower = min(max(preferred_lower, feasible_lower_minimum), feasible_lower_maximum)
    upper = lower + maximum_span
    return {
        "schema_version": 1,
        "policy_id": policy_id,
        "minimum_pitch": lower,
        "maximum_pitch": upper,
        "target_span_semitones": target_span,
        "maximum_span_semitones": maximum_span,
        "preferred_pitch_range": [preferred_lower, preferred_lower + maximum_span],
        "melody_containing_lower_bound_range": [
            containing_lower_minimum,
            containing_lower_maximum,
        ],
        "physical_lower_bound_range": [21, 108 - maximum_span],
        "feasible_lower_bound_range": [
            feasible_lower_minimum,
            feasible_lower_maximum,
        ],
        "lower_bound_projection_semitones": lower - preferred_lower,
        "reachable_after_melody": True,
        "main_melody_progress": progress,
    }


def validate_material_pitch_range(
    pitches_by_material: Mapping[str, Sequence[int]],
    allowed_pitch_range: Mapping[str, Any],
) -> None:
    """新規旋律素材が確定済み共通音域を拡張しないことを検査する。"""

    try:
        lower = int(allowed_pitch_range["minimum_pitch"])
        upper = int(allowed_pitch_range["maximum_pitch"])
    except (KeyError, TypeError, ValueError) as error:
        raise WholeScoreLiveRunError("allowed pitch range is invalid") from error
    if not 21 <= lower <= upper <= 108:
        raise WholeScoreLiveRunError("allowed pitch range is invalid")
    outside = [
        (material_id, pitch)
        for material_id, pitches in pitches_by_material.items()
        for pitch in pitches
        if not lower <= pitch <= upper
    ]
    if outside:
        raise WholeScoreLiveRunError(
            "melody transition exceeds the shared allowed pitch range"
        )


def _relative_register_bin(value: float) -> int:
    for index, boundary in enumerate((-12, -7, -3, 3, 7, 12)):
        if value < boundary:
            return index
    return 6


def measure_register_diagnostic(
    plan: PiecePlan,
    score: ScoreSpec,
    target: dict[str, Any],
) -> dict[str, Any]:
    """最終譜面を全leafへ展開し、参照音域記述子と同じ母集団で測る。"""

    pitches_by_material = {
        material.material_id: [note.pitch for note in material.notes]
        for material in score.materials
    }
    pitches = _expanded_pitch_values(plan, pitches_by_material)
    progress = measure_register_progress(plan, pitches_by_material)
    median_pitch = progress["median_pitch"]
    counts = [0] * 7
    for pitch in pitches:
        counts[_relative_register_bin(pitch - median_pitch)] += 1
    relative_actual = [count / len(pitches) for count in counts]
    pitch_range_actual = [min(progress["pitch_span_semitones"] / 88, 1.0)]
    try:
        pitch_target = target["pitch_range"]
        relative_target = target["relative_register_diagnostic"]
        pitch_center = [float(value) for value in pitch_target["neighborhood_center"]]
        pitch_radius = float(pitch_target["neighborhood_radius"])
        relative_center = [
            float(value) for value in relative_target["neighborhood_center"]
        ]
        relative_radius = float(relative_target["neighborhood_radius"])
    except (KeyError, TypeError, ValueError) as error:
        raise WholeScoreLiveRunError("register diagnostic target is invalid") from error
    pitch_distance = _metric_distance("scalar", pitch_range_actual, pitch_center)
    relative_distance = _metric_distance(
        "distribution", relative_actual, relative_center
    )
    return {
        "schema_version": 1,
        "population": "piece_plan_leaf_expanded_score_notes",
        **progress,
        "relative_register_distribution": relative_actual,
        "pitch_range": {
            "actual": pitch_range_actual,
            "neighborhood_center": pitch_center,
            "neighborhood_radius": pitch_radius,
            "neighborhood_center_distance": pitch_distance,
            "within_neighborhood": pitch_distance <= pitch_radius,
        },
        "relative_register": {
            "actual": relative_actual,
            "neighborhood_center": relative_center,
            "neighborhood_radius": relative_radius,
            "neighborhood_center_distance": relative_distance,
            "within_neighborhood": relative_distance <= relative_radius,
        },
    }


def validate_whole_score_source_plan(plan: PiecePlan) -> dict[str, Any]:
    """PiecePlanが曲全体生成器の早期終止と遷移契約を満たすか検査する。"""
    leaves = _ordered_leaves(plan)
    if not leaves:
        raise WholeScoreLiveRunError("source PiecePlan has no leaves")
    final_leaf = leaves[-1]
    material_counts: dict[str, int] = {}
    for leaf in leaves:
        if leaf.score_material_id is None:
            raise WholeScoreLiveRunError("source PiecePlan leaf has no material")
        material_counts[leaf.score_material_id] = (
            material_counts.get(leaf.score_material_id, 0) + 1
        )
    if (
        final_leaf.role != "release"
        or final_leaf.derived_from is not None
        or final_leaf.score_material_id is None
        or material_counts[final_leaf.score_material_id] != 1
    ):
        raise WholeScoreLiveRunError(
            "source PiecePlan final leaf must be a dedicated release material"
        )
    transition_indexes = [index for index, leaf in enumerate(leaves) if leaf.role == "transition"]
    if any(index in {0, len(leaves) - 1} for index in transition_indexes):
        raise WholeScoreLiveRunError("source PiecePlan transition cannot be first or last")
    if any(
        material_counts[leaves[index].score_material_id] != 1
        for index in transition_indexes
    ):
        raise WholeScoreLiveRunError(
            "source PiecePlan transition material must be dedicated to one leaf"
        )
    return {
        "status": "pass",
        "leaf_count": len(leaves),
        "transition_count": len(transition_indexes),
        "final_leaf_id": final_leaf.node_id,
        "final_material_id": final_leaf.score_material_id,
    }


def _validate_legacy_prefix_provenance(
    project_root: Path,
    prefix_path: Path,
    provenance_path: Path,
) -> str:
    provenance = _read_json(provenance_path)
    try:
        target_version = str(provenance["target_version"])
        prefix_sha256 = str(provenance["prepared_texture_prefix_sha256"])
        target_record = provenance["source_prompt_target"]
        spec_record = provenance["source_run_spec"]
        target_path = Path(str(target_record["path"]))
        spec_path = Path(str(spec_record["path"]))
    except (KeyError, TypeError, ValueError) as error:
        raise WholeScoreLiveRunError("legacy prefix provenance is invalid") from error
    if provenance.get("schema_version") != 1 or _target_version_number(
        target_version
    ) >= 4:
        raise WholeScoreLiveRunError("legacy prefix provenance is invalid")
    if prefix_sha256 != sha256_file(prefix_path):
        raise WholeScoreLiveRunError("legacy prefix provenance is invalid")
    for path, record in ((target_path, target_record), (spec_path, spec_record)):
        resolved = path if path.is_absolute() else project_root / path
        if not resolved.is_file() or record.get("sha256") != sha256_file(resolved):
            raise WholeScoreLiveRunError("legacy prefix provenance is invalid")
    saved_target = _read_json(
        target_path if target_path.is_absolute() else project_root / target_path
    )
    if saved_target.get("target_version") != target_version:
        raise WholeScoreLiveRunError("legacy prefix provenance is invalid")
    return target_version


def prepare_whole_score_live_run(
    project_root: Path,
    run_dir: Path,
    *,
    source: WholeScorePreparedSource | None = None,
) -> dict[str, Any]:
    """保存済みPiecePlanの来歴を再検証し、外部実走入力を不変化する。"""

    project_root = Path(project_root).resolve()
    run_dir = Path(run_dir).resolve()
    if source is None:
        source = WholeScorePreparedSource(
            run_root=_SOURCE_RUN,
            use_legacy_v7_frequency=True,
            source_smf_relative=Path("outputs/final.mid"),
        )
    source_root = (
        source.run_root.resolve()
        if source.run_root.is_absolute()
        else (project_root / source.run_root).resolve()
    )
    plan_path = source_root / "outputs/piece-plan.dsl"
    target_path = source_root / "inputs/prompt-target.json"
    resolved_path = source_root / "inputs/resolved-request.json"
    source_spec_path = source_root / "run-spec.json"
    source_state_path = source_root / "run-state.json"
    source_spec = _read_json(source_spec_path)
    source_state = _read_json(source_state_path)
    if source_state.get("status") == "failed":
        raise WholeScoreLiveRunError("source run has failed")
    try:
        plan_step = source_state["steps"]["piece-plan"]
        recorded_plan_hash = plan_step["outputs"]["sha256"]
    except (KeyError, TypeError) as error:
        raise WholeScoreLiveRunError("source PiecePlan provenance is missing") from error
    if plan_step.get("status") != "completed":
        raise WholeScoreLiveRunError("source PiecePlan step is not completed")
    if recorded_plan_hash != sha256_file(plan_path):
        raise WholeScoreLiveRunError("source PiecePlan hash mismatch")
    plan = parse_piece_plan(plan_path.read_text(encoding="utf-8"))
    quality = evaluate_generic_piece_plan_quality(plan)
    if not quality.get("passes"):
        raise WholeScoreLiveRunError("source PiecePlan generic quality failed")
    compatibility = validate_whole_score_source_plan(plan)

    saved_target = _read_json(target_path)
    generation_intent_path = source_root / "inputs/generation-intent.json"
    generation_intent = (
        _read_json(generation_intent_path)
        if generation_intent_path.is_file()
        else None
    )
    saved_intent_metadata = saved_target.get("generation_intent")
    if (generation_intent is None) != (saved_intent_metadata is None):
        raise WholeScoreLiveRunError("source generation intent provenance is incomplete")
    if generation_intent is not None and (
        not isinstance(saved_intent_metadata, Mapping)
        or saved_intent_metadata.get(
            "source_sha256"
        )
        != sha256_json(generation_intent)
    ):
        raise WholeScoreLiveRunError("source generation intent hash mismatch")
    resolved = _read_json(resolved_path)
    if source_spec.get("prompt_target_sha256") != sha256_json(saved_target):
        raise WholeScoreLiveRunError("source prompt target hash mismatch")
    reference_dir = project_root / ".appendix/reference-profile-v1"
    control_dir = project_root / ".appendix/control-reference-baseline-v3"
    rebuilt = build_reference_generation_target(
        resolved,
        reference_dir=reference_dir,
        control_dir=control_dir,
    )
    saved_version = saved_target.get("target_version")
    legacy_attack_frequency_target = None
    if (
        saved_version
        in {
            "reference-generation-target-v2",
            "reference-generation-target-v3",
            "reference-generation-target-v4",
            "reference-generation-target-v5",
            "reference-generation-target-v6",
        }
        and isinstance(resolved.get("controls"), dict)
        and "attack_frequency" in resolved["controls"]
    ):
        legacy_attack_frequency_target = build_reference_generation_target(
            resolved,
            reference_dir=reference_dir,
            control_dir=control_dir,
            legacy_reference_attack_frequency=True,
        ).prompt_target
    reference_comparable_target = _prompt_target_for_saved_version(
        rebuilt.prompt_target,
        saved_version,
        legacy_attack_frequency_target=legacy_attack_frequency_target,
    )
    comparable_target = (
        merge_generation_intent(reference_comparable_target, generation_intent)
        if generation_intent is not None
        else reference_comparable_target
    )
    if comparable_target != saved_target:
        raise WholeScoreLiveRunError("rebuilt prompt target differs from saved input")
    reference_current_target = (
        reference_comparable_target
        if source.register_enforcement_mode
        == "diagnostic_only_lower_bound_experiment"
        and saved_target.get("target_version") == "reference-generation-target-v4"
        else rebuilt.prompt_target
    )
    current_target = (
        merge_generation_intent(reference_current_target, generation_intent)
        if generation_intent is not None
        else reference_current_target
    )
    descriptor_ids = _descriptor_ids(current_target)

    attack_frequency = (
        _legacy_v7_attack_frequency()
        if source.use_legacy_v7_frequency
        else _semantic_attack_frequency(current_target)
    )

    generation_paths = {
        "piece_plan": plan_path,
        "prompt_target": target_path,
        "resolved_request": resolved_path,
        "source_run_spec": source_spec_path,
        "source_run_state": source_state_path,
        "reference_manifest": Path(".appendix/reference-profile-v1/manifest.json"),
        "reference_summary": Path(".appendix/reference-profile-v1/summary.json"),
        "control_manifest": Path(".appendix/control-reference-baseline-v3/manifest.json"),
        "control_summary": Path(".appendix/control-reference-baseline-v3/summary.json"),
    }
    if generation_intent is not None:
        generation_paths["generation_intent"] = generation_intent_path
    if source.harmonic_collection_path is not None:
        prepared_harmony = (
            source.harmonic_collection_path.resolve()
            if source.harmonic_collection_path.is_absolute()
            else (project_root / source.harmonic_collection_path).resolve()
        )
        generation_paths["prepared_harmonic_collection"] = prepared_harmony
    if source.melody_collection_path is not None:
        prepared_melody = (
            source.melody_collection_path.resolve()
            if source.melody_collection_path.is_absolute()
            else (project_root / source.melody_collection_path).resolve()
        )
        generation_paths["prepared_melody_collection"] = prepared_melody
    if source.texture_budget_path is not None:
        prepared_texture_budget = (
            source.texture_budget_path.resolve()
            if source.texture_budget_path.is_absolute()
            else (project_root / source.texture_budget_path).resolve()
        )
        generation_paths["prepared_texture_budget"] = prepared_texture_budget
    if source.texture_collection_prefix_path is not None:
        prepared_texture_prefix = (
            source.texture_collection_prefix_path.resolve()
            if source.texture_collection_prefix_path.is_absolute()
            else (project_root / source.texture_collection_prefix_path).resolve()
        )
        generation_paths["prepared_texture_prefix"] = prepared_texture_prefix
    if source.texture_prefix_provenance_path is not None:
        texture_prefix_provenance = (
            source.texture_prefix_provenance_path.resolve()
            if source.texture_prefix_provenance_path.is_absolute()
            else (project_root / source.texture_prefix_provenance_path).resolve()
        )
        generation_paths["texture_prefix_provenance"] = (
            texture_prefix_provenance
        )
    else:
        texture_prefix_provenance = None
    register_exception_provenance: dict[str, Any] | None = None
    register_exception_failure: Path | None = None
    legacy_prefix_provenance: Path | None = None
    register_target_version = str(current_target.get("target_version", ""))
    legacy_verified = False
    if source.register_enforcement_mode == "legacy_prefix_diagnostic_only":
        if (
            source.texture_collection_prefix_path is None
            or source.legacy_prefix_provenance_path is None
        ):
            raise WholeScoreLiveRunError("legacy prefix provenance is missing")
        legacy_prefix_provenance = (
            source.legacy_prefix_provenance_path.resolve()
            if source.legacy_prefix_provenance_path.is_absolute()
            else (project_root / source.legacy_prefix_provenance_path).resolve()
        )
        register_target_version = _validate_legacy_prefix_provenance(
            project_root,
            prepared_texture_prefix,
            legacy_prefix_provenance,
        )
        legacy_verified = True
        generation_paths["legacy_prefix_provenance"] = legacy_prefix_provenance
    if (
        source.register_enforcement_mode
        == "diagnostic_only_lower_bound_experiment"
    ):
        if source.register_exception_provenance_path is None:
            raise WholeScoreLiveRunError(
                "register exception provenance is missing"
            )
        if (
            source.harmonic_collection_path is None
            or source.melody_collection_path is None
            or source.texture_budget_path is None
            or source.texture_batch_maximum_event_count is None
        ):
            raise WholeScoreLiveRunError(
                "register exception provenance inputs are missing"
            )
        provenance_path = (
            source.register_exception_provenance_path.resolve()
            if source.register_exception_provenance_path.is_absolute()
            else (project_root / source.register_exception_provenance_path).resolve()
        )
        declared = _read_json(provenance_path)
        try:
            failure_record = declared["source_failure"]
            failure_record_path = Path(str(failure_record["path"]))
        except (KeyError, TypeError, ValueError) as error:
            raise WholeScoreLiveRunError(
                "register exception provenance is invalid"
            ) from error
        register_exception_failure = (
            failure_record_path
            if failure_record_path.is_absolute()
            else (project_root / failure_record_path).resolve()
        )
        expected = build_register_exception_provenance(
            project_root,
            policy_id=str(declared.get("policy_id", "")),
            plan_path=plan_path,
            prompt_target=current_target,
            harmonic_collection_path=prepared_harmony,
            melody_collection_path=prepared_melody,
            texture_budget_path=prepared_texture_budget,
            failure_path=register_exception_failure,
            texture_batch_maximum_event_count=(
                source.texture_batch_maximum_event_count
            ),
        )
        if declared != expected:
            raise WholeScoreLiveRunError(
                "register exception provenance differs from deterministic rebuild"
            )
        register_exception_provenance = expected
        generation_paths["register_exception_provenance"] = provenance_path
        generation_paths["register_exception_failure"] = (
            register_exception_failure
        )
    register_enforcement = resolve_register_enforcement(
        requested_mode=source.register_enforcement_mode,
        target_version=register_target_version,
        has_prepared_prefix=source.texture_collection_prefix_path is not None,
        has_verified_legacy_provenance=legacy_verified,
    )
    if register_exception_provenance is not None:
        counterfactual = register_exception_provenance[
            "entry_counterfactual"
        ]
        register_enforcement.update(
            {
                "normal_allowed_pitch_range": counterfactual[
                    "normal_allowed_pitch_range"
                ],
                "experimental_allowed_pitch_range": counterfactual[
                    "experimental_allowed_pitch_range"
                ],
                "provenance_sha256": sha256_json(
                    register_exception_provenance
                ),
            }
        )
    evaluation_paths = {
        "reference_profiles": Path(".appendix/reference-profile-v1/files.jsonl"),
        "capability_v26_smf": Path(
            ".appendix/long-form-runs/20260818-long-form-v26-ending-voice-leading/final.mid"
        ),
        "multiscale_v8_smf": Path(
            ".appendix/multiscale-calibration-run-v8/outputs/final.mid"
        ),
    }
    if source.source_smf_relative is not None:
        source_smf = source_root / source.source_smf_relative
        if source_smf.is_file():
            evaluation_paths["source_v7_a2_smf"] = source_smf
    input_hashes = {
        "generation_inputs": {
            name: _file_record(project_root, path)
            for name, path in generation_paths.items()
        },
        "evaluation_inputs": {
            name: _file_record(project_root, path)
            for name, path in evaluation_paths.items()
        },
    }
    implementation_hashes = {
        path.as_posix(): sha256_file(project_root / path) for path in _IMPLEMENTATIONS
    }
    maximum_external_calls = (
        DEFAULT_MAXIMUM_EXTERNAL_CALLS
        if source.maximum_external_calls is None
        else source.maximum_external_calls
    )
    if not 1 <= maximum_external_calls <= MAXIMUM_EXTERNAL_CALLS:
        raise WholeScoreLiveRunError("source external call ceiling is invalid")
    if (
        source.texture_batch_maximum_event_count is not None
        and source.texture_batch_maximum_event_count <= 0
    ):
        raise WholeScoreLiveRunError("texture batch event ceiling is invalid")
    if (
        source.texture_batch_maximum_count is not None
        and source.texture_batch_maximum_count <= 0
    ):
        raise WholeScoreLiveRunError("texture batch count ceiling is invalid")
    if source.generation_profile_id not in {
        "legacy-v1",
        "prepared-continuation-v1",
        "stable-staged-v1",
        "stable-staged-v2",
        "stable-staged-v3",
    }:
        raise WholeScoreLiveRunError("generation profile is invalid")
    if source.velocity_policy_id not in {
        "legacy-unison-v1",
        "foreground-accompaniment-harmony-shape-v1",
    }:
        raise WholeScoreLiveRunError("velocity policy is invalid")
    if source.key_release_unreachable_policy not in {"raise", "nearest_unfit"}:
        raise WholeScoreLiveRunError("key release unreachable policy is invalid")
    stable_batch_count = (
        6 if source.generation_profile_id == "stable-staged-v3" else 4
    )
    if source.generation_profile_id in {
        "stable-staged-v1",
        "stable-staged-v2",
        "stable-staged-v3",
    } and (
        maximum_external_calls != 9
        or source.texture_batch_maximum_event_count != 400
        or source.texture_batch_maximum_count != stable_batch_count
        or source.velocity_policy_id
        != "foreground-accompaniment-harmony-shape-v1"
        or source.key_release_unreachable_policy != "nearest_unfit"
    ):
        raise WholeScoreLiveRunError("stable generation profile settings conflict")
    if source.generation_profile_id == "prepared-continuation-v1" and (
        source.harmonic_collection_path is None
        or source.melody_collection_path is None
        or source.texture_budget_path is None
        or source.texture_collection_prefix_path is None
        or source.texture_prefix_provenance_path is None
        or source.maximum_external_calls is None
        or source.texture_batch_maximum_event_count is None
        or source.texture_batch_maximum_count is None
        or source.velocity_policy_id
        != "foreground-accompaniment-harmony-shape-v1"
        or source.key_release_unreachable_policy != "nearest_unfit"
        or source.register_enforcement_mode != "hard"
    ):
        raise WholeScoreLiveRunError(
            "prepared continuation profile settings conflict"
        )
    model_config = {
        **MODEL_CONFIG,
        "maximum_external_calls": maximum_external_calls,
    }
    spec = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_config": model_config,
        "input_hashes": input_hashes,
        "implementation_hashes": implementation_hashes,
        "descriptor_ids": list(descriptor_ids),
        "prompt_target_sha256": sha256_json(current_target),
        "attack_frequency": attack_frequency,
        "texture_batch_maximum_event_count": (
            source.texture_batch_maximum_event_count
        ),
        "texture_batch_maximum_count": source.texture_batch_maximum_count,
        "generation_profile_id": source.generation_profile_id,
        "velocity_policy_id": source.velocity_policy_id,
        "key_release_unreachable_policy": source.key_release_unreachable_policy,
        "register_enforcement": register_enforcement,
        "texture_placement_policy": (
            "search-aware-onset-zone-v8"
            if source.generation_profile_id
            in {
                "prepared-continuation-v1",
                "stable-staged-v2",
                "stable-staged-v3",
            }
            else "onset-feasible-zone-v7"
            if _target_version_number(str(current_target["target_version"])) >= 5
            and register_enforcement["effective_mode"] == "hard"
            else "bounded-backtracking-v4"
        ),
    }
    store = RunStore(run_dir, max_calls=maximum_external_calls)
    store.initialize(spec)
    store.snapshot_file("inputs/piece-plan.dsl", plan_path)
    store.snapshot_json("inputs/prompt-target.json", current_target)
    store.snapshot_json("inputs/resolved-request.json", resolved)
    if generation_intent is not None:
        store.snapshot_json("inputs/generation-intent.json", generation_intent)
    if source.harmonic_collection_path is not None:
        store.snapshot_file(
            "inputs/prepared-harmonic-collection.dsl",
            prepared_harmony,
        )
    if source.melody_collection_path is not None:
        store.snapshot_file(
            "inputs/prepared-melody-collection.dsl",
            prepared_melody,
        )
    if source.texture_budget_path is not None:
        store.snapshot_file(
            "inputs/prepared-texture-budget.json",
            prepared_texture_budget,
        )
    if source.texture_collection_prefix_path is not None:
        store.snapshot_file(
            "inputs/prepared-texture-prefix.dsl",
            prepared_texture_prefix,
        )
    if texture_prefix_provenance is not None:
        store.snapshot_file(
            "inputs/prepared-texture-prefix-provenance.json",
            texture_prefix_provenance,
        )
    if legacy_prefix_provenance is not None:
        store.snapshot_file(
            "inputs/legacy-prefix-provenance.json",
            legacy_prefix_provenance,
        )
    if register_exception_provenance is not None:
        store.snapshot_json(
            "inputs/register-exception-provenance.json",
            register_exception_provenance,
        )
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "prepared",
        "source_run_status": source_state["status"],
        "source_plan_compatibility": compatibility,
        "prompt_target_rebuilt": True,
        "descriptor_count": len(descriptor_ids),
        "maximum_external_calls": maximum_external_calls,
        "harmony_source": (
            "prepared" if source.harmonic_collection_path is not None else "external"
        ),
    }
    atomic_write_json(run_dir / "manifest.json", result)
    return result


def _render_prompt(template: str, context: dict[str, Any]) -> str:
    placeholder = "{{STAGE_CONTEXT_JSON}}"
    if template.count(placeholder) != 1:
        raise WholeScoreLiveRunError("stage prompt must contain one context placeholder")
    serialized = json.dumps(context, ensure_ascii=False, indent=2, sort_keys=True)
    return template.replace(placeholder, serialized)


def _response_source(response: dict[str, object], stage: str) -> str:
    source = response.get("composition_source")
    if not isinstance(source, str) or not source.strip():
        raise WholeScoreLiveRunError(f"{stage} response has no composition_source")
    return source


def _save_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def _save_stage_failure(
    run_dir: Path,
    step_id: str,
    source: str,
    error: Exception,
    input_hashes: dict[str, str],
) -> None:
    relative_path = Path("failures") / f"{step_id}.json"
    path = run_dir / relative_path
    failure = {
        "schema_version": (
            error.placement_schema_version
            if isinstance(error, _TextureBatchPlacementError)
            else 1
        ),
        "step_id": step_id,
        "error_type": type(error).__name__,
        "detail": str(error),
        "composition_source": source,
        "source_sha256": sha256_text(source),
    }
    if (
        isinstance(error, _TextureBatchPlacementError)
        and error.placement_schema_version >= 2
    ):
        failure["placement"] = error.placement
    atomic_write_json(path, failure)
    _run_store(run_dir).record_step(
        step_id,
        "failed",
        input_hashes,
        {
            "failure_path": str(relative_path),
            "failure_sha256": sha256_file(path),
        },
    )


def _creative_target_input_hashes(
    prompt_target: Mapping[str, object], step_id: str
) -> dict[str, str]:
    targets = creative_targets_for_step(prompt_target, step_id)
    return {"creative_targets": sha256_json(targets)} if targets else {}


def _harmonic_stage_context(
    plan: PiecePlan,
    prompt_target: dict[str, Any],
    attack_frequency: dict[str, Any],
) -> dict[str, Any]:
    context = build_whole_score_context(plan)
    leaves = _ordered_leaves(plan)
    material_indexes = {
        material_id: index
        for index, material_id in enumerate(context.material_ids, start=1)
    }
    relations = {
        target: source
        for relation in context.relations
        for target, source in relation.material_pairs
        if target != source
    }
    materials = []
    for material_id in context.material_ids:
        occurrences = [leaf for leaf in leaves if leaf.score_material_id == material_id]
        source = relations.get(material_id)
        materials.append(
            {
                "material_index": material_indexes[material_id],
                "occurrences": [
                    {
                        "role": leaf.role,
                        "harmonic_focus": leaf.harmonic_focus,
                        "duration_weight": leaf.duration_weight,
                        "position": leaves.index(leaf) + 1,
                    }
                    for leaf in occurrences
                ],
                "derived_from_material_index": (
                    material_indexes[source] if source is not None else None
                ),
                "is_transition": material_id in context.transition_material_ids,
                "is_release": material_id in context.release_material_ids,
            }
        )
    result = {
        "schema_version": 1,
        "piece": {
            "tonal_center": plan.tonal_center,
            "mode": plan.mode,
            "ending_intent": plan.ending_intent,
            "occurrence_count": len(leaves),
            "material_count": len(materials),
        },
        "material_order": materials,
        "controls": prompt_target["controls"],
        "semantic_targets": _specified_tonal_targets(prompt_target),
        "attack_frequency": {
            "target_groups_per_second": attack_frequency["target"],
            "candidate_range": [
                attack_frequency["candidate_minimum"],
                attack_frequency["candidate_maximum"],
            ],
            "group_tolerance_ms": attack_frequency["group_tolerance_ms"],
            "instruction": "reserve rhythmic capacity; do not add notes mechanically",
        },
    }
    creative_targets = creative_targets_for_step(prompt_target, "harmonic-collection")
    if creative_targets:
        result["creative_targets"] = creative_targets
    return result


def execute_harmony_stage(
    project_root: Path,
    run_dir: Path,
    runner: WholeScoreStageRunner,
) -> HarmonicSkeletonV0:
    """外部実走の和声集合を1回だけ生成し、早期停止条件を検査する。"""

    project_root = Path(project_root).resolve()
    run_dir = Path(run_dir).resolve()
    plan_source = (run_dir / "inputs/piece-plan.dsl").read_text(encoding="utf-8")
    plan = parse_piece_plan(plan_source)
    prompt_target = _read_json(run_dir / "inputs/prompt-target.json")
    attack_frequency = _run_attack_frequency(run_dir)
    prepared_path = run_dir / "inputs/prepared-harmonic-collection.dsl"
    if prepared_path.is_file():
        source = prepared_path.read_text(encoding="utf-8")
        stage_inputs = {
            "piece_plan": sha256_text(plan_source),
            "prompt_target": sha256_json(prompt_target),
            "prepared_harmonic_collection": sha256_text(source),
        }
        source_mode = "prepared"
    else:
        template = (project_root / _PROMPTS[0]).read_text(encoding="utf-8")
        context = _harmonic_stage_context(plan, prompt_target, attack_frequency)
        prompt = _render_prompt(template, context)
        _save_text(run_dir / "prompts/harmonic-collection.md", prompt)
        stage_inputs = {
            "piece_plan": sha256_text(plan_source),
            "prompt_target": sha256_json(prompt_target),
            "prompt": sha256_text(prompt),
            **_creative_target_input_hashes(prompt_target, "harmonic-collection"),
        }
        response = runner.run("harmonic-collection", prompt, stage_inputs)
        source = _response_source(response, "harmonic collection")
        _save_text(run_dir / "responses/harmonic-collection.dsl", source)
        source_mode = "external"
    raw_drafts = parse_harmonic_collection(source)
    raw_skeleton = assemble_whole_score_skeleton(
        "reference-v7-a2-live-v1",
        plan,
        tuple((item.length_units, item.draft) for item in raw_drafts),
    )
    raw_capacity = maximum_attack_group_capacity(plan, raw_skeleton)
    atomic_write_json(run_dir / "outputs/raw-attack-capacity.json", raw_capacity)
    validate_early_ending(plan, raw_skeleton)
    minimum_attack_group_count = int(
        attack_frequency["minimum_attack_group_count"]
    )
    insufficient = (
        raw_capacity["maximum_attack_group_count"] < minimum_attack_group_count
    )
    if insufficient and source_mode == "prepared":
        drafts = raw_drafts
        normalization = _harmonic_capacity_diagnostic(
            plan,
            raw_drafts,
            raw_drafts,
            status="rejected_prepared",
            minimum_attack_group_count=minimum_attack_group_count,
            reserve_ending_hold=True,
        )
    else:
        drafts, normalization = normalize_harmonic_capacity(
            plan,
            raw_drafts,
            minimum_attack_group_count=minimum_attack_group_count,
        )
    skeleton = assemble_whole_score_skeleton(
        "reference-v7-a2-live-v1",
        plan,
        tuple((item.length_units, item.draft) for item in drafts),
    )
    capacity = maximum_attack_group_capacity(plan, skeleton)
    ending = validate_early_ending(plan, skeleton)
    tonal_targets = _specified_tonal_targets(prompt_target)
    tonal_quality = evaluate_harmonic_skeleton_tonal_hierarchy(
        plan,
        skeleton,
        tonal_targets[0] if tonal_targets else {"status": "unverified_continuous_reference"},
    )
    atomic_write_json(run_dir / "outputs/tonal-hierarchy-harmony.json", tonal_quality)
    if not tonal_quality["passes"]:
        raise WholeScoreLiveRunError(
            f"harmonic stage tonal hierarchy failed: {tonal_quality['failures']}"
        )
    canonical = dump_harmonic_collection(drafts)
    normalization.update(
        {
            "raw_harmonic_collection_sha256": sha256_text(source),
            "effective_harmonic_collection_sha256": sha256_text(canonical),
        }
    )
    atomic_write_json(run_dir / "outputs/attack-capacity.json", capacity)
    atomic_write_json(
        run_dir / "outputs/harmonic-capacity-normalization.json",
        normalization,
    )
    if insufficient and source_mode == "prepared":
        raise WholeScoreLiveRunError("harmonic stage has insufficient attack capacity")
    if capacity["maximum_attack_group_count"] < minimum_attack_group_count:
        raise WholeScoreLiveRunError("harmonic stage has insufficient attack capacity")
    skeleton_source = dump_harmonic_skeleton(skeleton)
    _save_text(run_dir / "outputs/harmonic-collection.dsl", canonical)
    _save_text(run_dir / "outputs/harmonic-skeleton.dsl", skeleton_source)
    atomic_write_json(run_dir / "outputs/early-ending.json", ending)
    store = _run_store(run_dir)
    store.record_step(
        "harmonic-collection",
        "completed",
        stage_inputs,
        {
            "harmonic_skeleton_sha256": sha256_text(skeleton_source),
            "raw_capacity": raw_capacity,
            "effective_capacity": capacity,
            "capacity": capacity,
            "normalization_status": normalization["status"],
            "raw_harmonic_collection_sha256": normalization[
                "raw_harmonic_collection_sha256"
            ],
            "effective_harmonic_collection_sha256": normalization[
                "effective_harmonic_collection_sha256"
            ],
            "ending": ending,
            "source_mode": source_mode,
            "external_call_number": (
                runner.call_number if source_mode == "external" else None
            ),
            "material_count": len(skeleton.materials),
            "material_lengths": [item.length_units for item in skeleton.materials],
            "harmonic_event_counts": [len(item.harmonies) for item in skeleton.materials],
        },
    )
    return skeleton


def _melody_material_context(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    material_id: str,
    drafts: dict[str, Any],
) -> dict[str, Any]:
    context = build_whole_score_context(plan)
    material = next(item for item in skeleton.materials if item.material_id == material_id)
    indexes = {value: index for index, value in enumerate(context.material_ids, 1)}
    relation_source = next(
        (
            source
            for relation in context.relations
            for target, source in relation.material_pairs
            if target == material_id and target != source
        ),
        None,
    )
    item: dict[str, Any] = {
        "material_index": indexes[material_id],
        "length_units": material.length_units,
        "harmonies": [asdict(harmony) for harmony in material.harmonies],
        "maximum_event_count": min(
            64,
            6 * math.ceil(material.length_units / skeleton.divisions),
        ),
        "is_transition": material_id in context.transition_material_ids,
        "is_release": material_id in context.release_material_ids,
        "occurrences": [
            {
                "role": leaf.role,
                "harmonic_focus": leaf.harmonic_focus,
                "duration_weight": leaf.duration_weight,
            }
            for leaf in _ordered_leaves(plan)
            if leaf.score_material_id == material_id
        ],
        "relation_source_material_index": (
            indexes[relation_source] if relation_source is not None else None
        ),
    }
    if item["is_release"]:
        item.update(_ending_note_requirements(plan, skeleton))
    if relation_source is not None and relation_source in drafts:
        item["confirmed_relation_source"] = asdict(drafts[relation_source])
    transition = next(
        (item for item in context.transitions if item.material_id == material_id),
        None,
    )
    if transition is not None:
        previous = drafts[transition.previous_material_id]
        following = drafts[transition.next_material_id]
        previous_events = sorted(previous.events, key=lambda event: event.at_units)
        following_events = sorted(following.events, key=lambda event: event.at_units)
        item["connection_anchors"] = {
            "previous_foreground_voice": previous.foreground_voice,
            "previous_last_pitch": previous_events[-1].pitch,
            "next_foreground_voice": following.foreground_voice,
            "next_first_pitch": following_events[0].pitch,
            "maximum_boundary_leap_semitones": 5,
            "maximum_internal_leap_semitones": 5,
            "minimum_attack_count": 4,
            "forbid_same_pitch_restrike_at_next_boundary": True,
        }
    return item


def _execute_melody_collection(
    *,
    project_root: Path,
    run_dir: Path,
    runner: WholeScoreStageRunner,
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    prompt_target: dict[str, Any],
    material_ids: tuple[str, ...],
    known_drafts: dict[str, Any],
    step_id: str,
    register_progress: dict[str, Any] | None = None,
    allowed_pitch_range: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    template = (project_root / _PROMPTS[1]).read_text(encoding="utf-8")
    attack_frequency = _run_attack_frequency(run_dir)
    context = {
        "schema_version": 1,
        "piece": {
            "tonal_center": plan.tonal_center,
            "mode": plan.mode,
            "divisions": skeleton.divisions,
        },
        "material_order": [
            _melody_material_context(plan, skeleton, material_id, known_drafts)
            for material_id in material_ids
        ],
        "controls": prompt_target["controls"],
        "semantic_targets": _melody_semantic_targets(prompt_target),
        "attack_frequency": {
            "target_groups_per_second": attack_frequency["target"],
            "candidate_range": [
                attack_frequency["candidate_minimum"],
                attack_frequency["candidate_maximum"],
            ],
            "group_tolerance_ms": attack_frequency["group_tolerance_ms"],
        },
    }
    creative_targets = creative_targets_for_step(prompt_target, step_id)
    if creative_targets:
        context["creative_targets"] = creative_targets
    if register_progress is not None:
        context["register_progress"] = register_progress
    if allowed_pitch_range is not None:
        context["allowed_pitch_range"] = {
            "minimum_pitch": allowed_pitch_range["minimum_pitch"],
            "maximum_pitch": allowed_pitch_range["maximum_pitch"],
        }
    prompt = _render_prompt(template, context)
    _save_text(run_dir / f"prompts/{step_id}.md", prompt)
    response = runner.run(
        step_id,
        prompt,
        {
            "harmonic_skeleton": sha256_text(dump_harmonic_skeleton(skeleton)),
            "prompt_target": sha256_json(prompt_target),
            "prompt": sha256_text(prompt),
            **_creative_target_input_hashes(prompt_target, step_id),
        },
    )
    source = _response_source(response, step_id)
    _save_text(run_dir / f"responses/{step_id}.dsl", source)
    drafts = parse_melody_collection(source)
    if len(drafts) != len(material_ids):
        raise WholeScoreLiveRunError(f"{step_id} count does not match fixed order")
    canonical = dump_melody_collection(drafts)
    _save_text(run_dir / f"outputs/{step_id}.dsl", canonical)
    _run_store(run_dir).record_step(
        step_id,
        "completed",
        {
            "harmonic_skeleton": sha256_text(dump_harmonic_skeleton(skeleton)),
            "prompt_target": sha256_json(prompt_target),
            "prompt": sha256_text(prompt),
            **_creative_target_input_hashes(prompt_target, step_id),
        },
        {
            "draft_sha256": sha256_text(canonical),
            "material_count": len(drafts),
            "external_call_number": runner.call_number,
        },
    )
    return drafts


def execute_melody_stages(
    project_root: Path,
    run_dir: Path,
    runner: WholeScoreStageRunner,
    skeleton: HarmonicSkeletonV0,
) -> ScorePayloadV0:
    """通常素材と接続素材を2巡で生成し、和声外音と接続を検査する。"""

    project_root = Path(project_root).resolve()
    run_dir = Path(run_dir).resolve()
    plan = parse_piece_plan(
        (run_dir / "inputs/piece-plan.dsl").read_text(encoding="utf-8")
    )
    prompt_target = _read_json(run_dir / "inputs/prompt-target.json")
    context = build_whole_score_context(plan)
    main_ids, transition_ids = melody_generation_order(context)
    prepared_path = run_dir / "inputs/prepared-melody-collection.dsl"
    known: dict[str, Any]
    if prepared_path.is_file():
        prepared_source = prepared_path.read_text(encoding="utf-8")
        prepared_drafts = parse_melody_collection(prepared_source)
        if len(prepared_drafts) != len(context.material_ids):
            raise WholeScoreLiveRunError(
                "prepared melody count does not match fixed order"
            )
        known = dict(zip(context.material_ids, prepared_drafts, strict=True))
        canonical = dump_melody_collection(prepared_drafts)
        _save_text(run_dir / "outputs/melody-collection-main.dsl", canonical)
        _run_store(run_dir).record_step(
            "melody-collection-main",
            "completed",
            {
                "harmonic_skeleton": sha256_text(dump_harmonic_skeleton(skeleton)),
                "prompt_target": sha256_json(prompt_target),
                "prepared_melody_collection": sha256_text(prepared_source),
            },
            {
                "draft_sha256": sha256_text(canonical),
                "material_count": len(prepared_drafts),
                "source_mode": "prepared",
                "external_call_number": None,
            },
        )
    else:
        known = {}
        main_drafts = _execute_melody_collection(
            project_root=project_root,
            run_dir=run_dir,
            runner=runner,
            plan=plan,
            skeleton=skeleton,
            prompt_target=prompt_target,
            material_ids=main_ids,
            known_drafts=known,
            step_id="melody-collection-main",
        )
        known.update(zip(main_ids, main_drafts, strict=True))
    register_target = _semantic_target_view(prompt_target, "register_envelope")[0]
    main_pitch_map = {
        material_id: [event.pitch for event in known[material_id].events]
        for material_id in main_ids
    }
    register_range = build_allowed_pitch_range(
        plan,
        main_pitch_map,
        register_target,
    )
    atomic_write_json(run_dir / "outputs/register-range.json", register_range)
    if prepared_path.is_file():
        validate_material_pitch_range(
            {
                material_id: [event.pitch for event in known[material_id].events]
                for material_id in transition_ids
            },
            register_range,
        )
    else:
        transition_drafts = (
            _execute_melody_collection(
                project_root=project_root,
                run_dir=run_dir,
                runner=runner,
                plan=plan,
                skeleton=skeleton,
                prompt_target=prompt_target,
                material_ids=transition_ids,
                known_drafts=known,
                step_id="melody-collection-transition",
                register_progress=register_range["main_melody_progress"],
                allowed_pitch_range=register_range,
            )
            if transition_ids
            else ()
        )
        validate_material_pitch_range(
            {
                material_id: [event.pitch for event in draft.events]
                for material_id, draft in zip(
                    transition_ids, transition_drafts, strict=True
                )
            },
            register_range,
        )
        known.update(zip(transition_ids, transition_drafts, strict=True))
    ordered = tuple(known[material_id] for material_id in context.material_ids)
    ordered, ending_normalization = normalize_melody_ending(
        plan,
        skeleton,
        ordered,
        allowed_pitch_range=(
            int(register_range["minimum_pitch"]),
            int(register_range["maximum_pitch"]),
        ),
    )
    validate_material_pitch_range(
        {
            material_id: [event.pitch for event in draft.events]
            for material_id, draft in zip(
                context.material_ids, ordered, strict=True
            )
        },
        register_range,
    )
    normalized_source = dump_melody_collection(ordered)
    _save_text(run_dir / "outputs/melody-collection.dsl", normalized_source)
    atomic_write_json(
        run_dir / "outputs/melody-ending-normalization.json",
        ending_normalization,
    )
    _run_store(run_dir).record_step(
        "melody-ending-normalization",
        "completed",
        {
            "harmonic_skeleton": sha256_text(dump_harmonic_skeleton(skeleton)),
            "source_melody_collection": sha256_text(
                dump_melody_collection(
                    tuple(known[material_id] for material_id in context.material_ids)
                )
            ),
        },
        {
            "normalized_melody_collection": sha256_text(normalized_source),
            "diagnostic": sha256_json(ending_normalization),
        },
    )
    payload = assemble_whole_score_melodies(
        "reference-v7-a2-live-v1",
        plan,
        skeleton,
        ordered,
    )
    ending = validate_melody_ending(plan, skeleton, payload)
    atomic_write_json(run_dir / "outputs/melody-ending.json", ending)
    provisional_score = apply_harmonic_skeleton(plan, payload, skeleton)
    dissonance = tuple(
        analyze_foreground_dissonance(provisional_score, material_id)
        for material_id in context.material_ids
    )
    atomic_write_json(
        run_dir / "outputs/foreground-dissonance.json",
        {"schema_version": 1, "items": [asdict(item) for item in dissonance]},
    )
    if any(not item.passes for item in dissonance):
        raise WholeScoreLiveRunError("melody has unsupported foreground dissonance")
    transitions = tuple(
        analyze_foreground_transition(
            plan,
            provisional_score,
            transition,
            minimum_transition_attacks=4,
            maximum_boundary_leap=5,
            maximum_internal_leap=5,
        )
        for transition in context.transitions
    )
    atomic_write_json(
        run_dir / "outputs/transition-assessments.json",
        {"schema_version": 1, "items": [asdict(item) for item in transitions]},
    )
    if any(not item.passes for item in transitions):
        raise WholeScoreLiveRunError("melody transition assessment failed")
    score_source = dump_score_spec(provisional_score)
    _save_text(run_dir / "outputs/melody-score-spec.dsl", score_source)
    return payload


def partition_texture_batches(
    materials: list[dict[str, Any]],
    *,
    maximum_event_count: int | None,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """素材順を保ち、伴奏event上限以下の最大連続batchへ分ける。"""

    if not materials:
        raise WholeScoreLiveRunError("texture batch materials must not be empty")
    if maximum_event_count is None:
        return (tuple(materials),)
    if maximum_event_count <= 0:
        raise WholeScoreLiveRunError("texture batch event ceiling is invalid")
    batches: list[tuple[dict[str, Any], ...]] = []
    current: list[dict[str, Any]] = []
    current_count = 0
    for material in materials:
        try:
            event_count = int(material["required_texture_event_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise WholeScoreLiveRunError("texture material budget is invalid") from error
        if event_count <= 0 or event_count > maximum_event_count:
            raise WholeScoreLiveRunError(
                "single texture material exceeds batch event ceiling"
            )
        if current and current_count + event_count > maximum_event_count:
            batches.append(tuple(current))
            current = []
            current_count = 0
        current.append(material)
        current_count += event_count
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _validate_texture_draft_budget(
    melody: Any,
    draft: Any,
    budget: dict[str, Any],
    *,
    maximum_group_size: int,
) -> dict[str, Any]:
    melody_counts: dict[int, int] = {}
    for note in melody.notes:
        melody_counts[note.at_units] = melody_counts.get(note.at_units, 0) + 1
    texture_counts: dict[int, int] = {}
    for event in draft.events:
        texture_counts[event.at_units] = texture_counts.get(event.at_units, 0) + 1
    onsets = sorted(melody_counts.keys() | texture_counts.keys())
    group_sizes = [
        melody_counts.get(onset, 0) + texture_counts.get(onset, 0)
        for onset in onsets
    ]
    size_counts = {
        "one": sum(size == 1 for size in group_sizes),
        "two": sum(size == 2 for size in group_sizes),
        "three": sum(size == 3 for size in group_sizes),
        "four_or_more": sum(size >= 4 for size in group_sizes),
    }
    actual = {
        "required_texture_event_count": len(draft.events),
        "combined_attack_group_count": len(onsets),
        "combined_note_event_count": len(melody.notes) + len(draft.events),
        "attack_size_counts": size_counts,
        "maximum_group_size": max(group_sizes, default=0),
    }
    expected = {
        "required_texture_event_count": budget["required_texture_event_count"],
        "combined_attack_group_count": budget["combined_attack_group_count"],
        "combined_note_event_count": budget["combined_note_event_count"],
        "attack_size_counts": budget["attack_size_counts"],
    }
    if any(actual[key] != value for key, value in expected.items()):
        raise WholeScoreLiveRunError(
            "texture budget: draft structure does not match material budget"
        )
    if actual["maximum_group_size"] > maximum_group_size:
        raise WholeScoreLiveRunError(
            "texture budget: draft exceeds maximum attack group size"
        )
    return {"expected": expected, "actual": actual, "matches_budget": True}


def _validate_release_texture_ending(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    material: Any,
    draft: Any,
) -> dict[str, Any]:
    requirements = _ending_note_requirements(plan, skeleton)
    required_attack = int(requirements["required_final_attack_units"])
    required_duration = int(requirements["required_final_duration_units"])
    final_attack = max((event.at_units for event in draft.events), default=None)
    if final_attack != required_attack:
        raise WholeScoreLiveRunError(
            "texture ending does not match the shared ending attack"
        )
    final_events = tuple(
        event for event in draft.events if event.at_units == required_attack
    )
    final_degrees = {event.degree for event in final_events}
    if len(final_events) < 2 or not {"root", "fifth"}.issubset(final_degrees) or any(
        event.duration_units != required_duration
        or event.at_units + event.duration_units != material.length_units
        for event in final_events
    ):
        raise WholeScoreLiveRunError(
            "texture ending does not satisfy the shared ending chord"
        )
    if any(
        event.at_units < required_attack
        and event.at_units + event.duration_units > required_attack
        for event in draft.events
    ):
        raise WholeScoreLiveRunError(
            "texture ending carries over a prior event"
        )
    return {
        "status": "pass",
        "material_id": material.material_id,
        "required_final_attack_units": required_attack,
        "required_final_duration_units": required_duration,
        "minimum_required_final_new_accompaniment_attack_count": 2,
        "actual_final_new_accompaniment_attack_count": len(final_events),
        "final_degrees": sorted(final_degrees),
        "prior_carryover_count": 0,
    }


def _validate_partial_texture_batches(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    melodies: ScorePayloadV0,
    texture_budget: dict[str, Any],
    drafts_by_material: dict[str, Any],
    allowed_pitch_range: tuple[int, int] | None,
    placement_policy: str = "bounded-backtracking-v4",
    score_case_id: str | None = None,
) -> dict[str, Any]:
    base_score = apply_harmonic_skeleton(plan, melodies, skeleton)
    payloads = {item.material_id: item for item in melodies.materials}
    budget_materials = {
        item["material_id"]: item for item in texture_budget["materials"]
    }
    score = base_score
    placements = []
    low_spacing: dict[str, int] = {}
    ending: dict[str, Any] | None = None
    release_material_ids = set(build_whole_score_context(plan).release_material_ids)
    for material_index, material in enumerate(skeleton.materials, 1):
        draft = drafts_by_material.get(material.material_id)
        if draft is None:
            continue
        budget = budget_materials[material.material_id]
        if material.material_id in release_material_ids:
            ending = _validate_release_texture_ending(
                plan, skeleton, material, draft
            )
        _validate_texture_draft_budget(
            payloads[material.material_id],
            draft,
            budget,
            maximum_group_size=texture_budget["score_spec_target"][
                "maximum_group_size"
            ],
        )
        preplacement = texture_preplacement_violations(
            draft,
            material.harmonies,
            payloads[material.material_id].notes,
            allowed_pitch_range=allowed_pitch_range,
        )
        if preplacement["event_feasibility"]:
            raise WholeScoreLiveRunError(
                "texture batch violates event feasibility contract: "
                f"{preplacement['event_feasibility']}"
            )
        if preplacement["onset_capacity"]:
            raise WholeScoreLiveRunError(
                "texture batch exceeds onset capacity contract: "
                f"{preplacement['onset_capacity']}"
            )
        target = next(
            item for item in score.materials if item.material_id == material.material_id
        )
        texture_case_id = (
            f"{score_case_id}-{material_index:03d}"
            if score_case_id is not None
            else f"batch-{material.material_id}"
        )
        placement, scored = assemble_texture(
            texture_case_id,
            plan,
            score,
            target,
            material.harmonies,
            payloads[material.material_id].notes,
            draft,
            maximum_event_count=budget["maximum_texture_event_count"],
            placement_policy=placement_policy,
            allowed_pitch_range=allowed_pitch_range,
        )
        if placement.status != "placed":
            raise _TextureBatchPlacementError(material.material_id, placement)
        score = replace(
            score,
            materials=tuple(
                scored if item.material_id == material.material_id else item
                for item in score.materials
            ),
        )
        validate_score_spec(plan, score)
        violations = full_low_spacing_violations(scored)
        if violations:
            raise WholeScoreLiveRunError("texture batch has low spacing violations")
        low_spacing[material.material_id] = violations
        placements.append(asdict(placement))
    measurement = measure_score_texture_budget(plan, score, texture_budget)
    measured = {item["material_id"]: item for item in measurement["materials"]}
    if any(
        not measured[material_id]["matches_budget"]
        for material_id in drafts_by_material
    ):
        raise WholeScoreLiveRunError("texture batch material does not match budget")
    harmony = tuple(
        analyze_material_harmony(
            score,
            material_id,
            low_pitch_boundary=LOW_PITCH_BOUNDARY,
            minimum_low_spacing_semitones=MINIMUM_LOW_SPACING_SEMITONES,
        )
        for material_id in drafts_by_material
    )
    if any(not item.passes for item in harmony):
        raise WholeScoreLiveRunError("texture batch has invalid material harmony")
    dissonance = tuple(
        analyze_foreground_dissonance(score, material_id)
        for material_id in drafts_by_material
    )
    if any(not item.passes for item in dissonance):
        raise WholeScoreLiveRunError(
            "texture batch has unsupported foreground dissonance"
        )
    whole_score_quality: dict[str, Any] | None = None
    if set(drafts_by_material) == set(build_whole_score_context(plan).material_ids):
        whole_score_quality = evaluate_generic_score_quality(plan, score)
        if not whole_score_quality["passes"]:
            raise WholeScoreLiveRunError(
                "texture batch completed an invalid generic score"
            )
    score_source = dump_score_spec(score)
    result = {
        "schema_version": (
            4
            if placement_policy == "search-aware-onset-zone-v8"
            else 3
            if placement_policy == "onset-feasible-zone-v7"
            else 2
            if placement_policy == "bidirectional-low-spacing-v6"
            else 1
        ),
        "material_ids": list(drafts_by_material),
        "material_count": len(drafts_by_material),
        "temporary_score_spec_sha256": sha256_text(score_source),
        "placements": placements,
        "low_spacing_violations": low_spacing,
        "material_budget_matches": {
            material_id: measured[material_id]["matches_budget"]
            for material_id in drafts_by_material
        },
        "material_harmony": [asdict(item) for item in harmony],
        "material_dissonance": [asdict(item) for item in dissonance],
        "ending": ending,
        "whole_score_quality": whole_score_quality,
        "register_progress": measure_register_progress(
            plan,
            {
                material.material_id: [note.pitch for note in material.notes]
                for material in score.materials
            },
        ),
    }
    if placement_policy in {
        "bidirectional-low-spacing-v6",
        "onset-feasible-zone-v7",
        "search-aware-onset-zone-v8",
    }:
        result["placement_policy"] = placement_policy
    return result


def build_register_exception_counterfactual(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    melodies: ScorePayloadV0,
    texture_budget: dict[str, Any],
    drafts_by_material: dict[str, Any],
    *,
    normal_allowed_pitch_range: tuple[int, int],
    experimental_allowed_pitch_range: tuple[int, int],
) -> dict[str, Any]:
    """同じ伴奏応答を通常範囲と実験範囲で決定的に比較する。"""

    def evaluate(allowed_pitch_range: tuple[int, int]) -> dict[str, Any]:
        try:
            validation = _validate_partial_texture_batches(
                plan,
                skeleton,
                melodies,
                texture_budget,
                drafts_by_material,
                allowed_pitch_range,
            )
        except _TextureBatchPlacementError as error:
            return {
                "status": "failed",
                "error_type": type(error).__name__,
                "detail": str(error),
                "failure": {
                    "kind": "placement",
                    "material_id": error.material_id,
                    "placement": error.placement,
                },
            }
        except WholeScoreLiveRunError as error:
            return {
                "status": "failed",
                "error_type": type(error).__name__,
                "detail": str(error),
                "failure": {"kind": "validation", "detail": str(error)},
            }
        return {
            "status": "passed",
            "validation_sha256": sha256_json(validation),
            "material_ids": validation["material_ids"],
            "low_spacing_violations": validation["low_spacing_violations"],
        }

    return {
        "schema_version": 1,
        "normal_allowed_pitch_range": list(normal_allowed_pitch_range),
        "experimental_allowed_pitch_range": list(
            experimental_allowed_pitch_range
        ),
        "normal": evaluate(normal_allowed_pitch_range),
        "experimental": evaluate(experimental_allowed_pitch_range),
    }


def _record_existing_path(project_root: Path, path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    try:
        recorded = path.relative_to(project_root)
    except ValueError:
        recorded = path
    return _file_record(project_root, recorded)


def build_register_exception_provenance(
    project_root: Path,
    *,
    policy_id: str,
    plan_path: Path,
    prompt_target: dict[str, Any],
    harmonic_collection_path: Path,
    melody_collection_path: Path,
    texture_budget_path: Path,
    failure_path: Path,
    texture_batch_maximum_event_count: int,
) -> dict[str, Any]:
    """固定failureから音域下限実験の証跡を外部呼出しなしで再構築する。"""

    project_root = Path(project_root).resolve()
    plan_path = Path(plan_path).resolve()
    harmonic_collection_path = Path(harmonic_collection_path).resolve()
    melody_collection_path = Path(melody_collection_path).resolve()
    texture_budget_path = Path(texture_budget_path).resolve()
    failure_path = Path(failure_path).resolve()
    policy = _REGISTER_EXCEPTION_POLICIES.get(policy_id)
    if policy is None:
        raise WholeScoreLiveRunError("register exception policy is unknown")
    if texture_batch_maximum_event_count != int(
        policy["texture_batch_maximum_event_count"]
    ):
        raise WholeScoreLiveRunError(
            "register exception policy batch maximum differs"
        )
    actual_input_sha256 = {
        "piece_plan": sha256_file(plan_path),
        "prompt_target": sha256_json(prompt_target),
        "harmonic_collection": sha256_file(harmonic_collection_path),
        "melody_collection": sha256_file(melody_collection_path),
        "texture_budget": sha256_file(texture_budget_path),
        "source_failure": sha256_file(failure_path),
    }
    if actual_input_sha256 != policy["input_sha256"]:
        raise WholeScoreLiveRunError(
            "register exception policy input SHA-256 differs"
        )
    plan = parse_piece_plan(plan_path.read_text(encoding="utf-8"))
    harmonic_drafts = parse_harmonic_collection(
        harmonic_collection_path.read_text(encoding="utf-8")
    )
    skeleton = assemble_whole_score_skeleton(
        "register-lower-bound-experiment-v1",
        plan,
        tuple((item.length_units, item.draft) for item in harmonic_drafts),
    )
    melody_drafts = parse_melody_collection(
        melody_collection_path.read_text(encoding="utf-8")
    )
    context = build_whole_score_context(plan)
    if len(melody_drafts) != len(context.material_ids):
        raise WholeScoreLiveRunError(
            "register exception melody count does not match fixed order"
        )
    melodies = assemble_whole_score_melodies(
        "register-lower-bound-experiment-v1",
        plan,
        skeleton,
        melody_drafts,
    )
    texture_budget = _read_json(texture_budget_path)
    batches = partition_texture_batches(
        texture_budget["materials"],
        maximum_event_count=texture_batch_maximum_event_count,
    )
    first_batch_ids = tuple(item["material_id"] for item in batches[0])
    failure = _read_json(failure_path)
    source = failure.get("composition_source")
    if not isinstance(source, str) or not source.strip():
        raise WholeScoreLiveRunError(
            "register exception failure source is missing"
        )
    if failure.get("source_sha256") != sha256_text(source):
        raise WholeScoreLiveRunError(
            "register exception failure source hash mismatch"
        )
    texture_drafts = parse_texture_collection(source)
    if len(texture_drafts) != len(first_batch_ids):
        raise WholeScoreLiveRunError(
            "register exception failure count does not match first batch"
        )
    payloads = {item.material_id: item for item in melodies.materials}
    register_target = _semantic_target_view(prompt_target, "register_envelope")[0]
    main_ids, _ = melody_generation_order(context)
    normal_range_record = build_allowed_pitch_range(
        plan,
        {
            material_id: [note.pitch for note in payloads[material_id].notes]
            for material_id in main_ids
        },
        register_target,
    )
    normal_range = (
        int(normal_range_record["minimum_pitch"]),
        int(normal_range_record["maximum_pitch"]),
    )
    expected_normal_range = tuple(policy["normal_allowed_pitch_range"])
    experimental_range = tuple(policy["experimental_allowed_pitch_range"])
    if normal_range != expected_normal_range:
        raise WholeScoreLiveRunError(
            "register exception normal range differs from policy"
        )
    all_drafts = dict(zip(first_batch_ids, texture_drafts, strict=True))
    entry_material_id = str(policy["entry_material_id"])
    if entry_material_id not in all_drafts:
        raise WholeScoreLiveRunError(
            "register exception entry material is missing"
        )
    entry_counterfactual = build_register_exception_counterfactual(
        plan,
        skeleton,
        melodies,
        texture_budget,
        {entry_material_id: all_drafts[entry_material_id]},
        normal_allowed_pitch_range=normal_range,
        experimental_allowed_pitch_range=experimental_range,
    )
    if (
        entry_counterfactual["normal"]["status"] != "failed"
        or entry_counterfactual["experimental"]["status"] != "passed"
    ):
        raise WholeScoreLiveRunError(
            "register exception entry counterfactual does not justify the experiment: "
            + json.dumps(
                entry_counterfactual, ensure_ascii=False, sort_keys=True
            )
        )
    full_batch_counterfactual = build_register_exception_counterfactual(
        plan,
        skeleton,
        melodies,
        texture_budget,
        all_drafts,
        normal_allowed_pitch_range=normal_range,
        experimental_allowed_pitch_range=experimental_range,
    )
    full_experimental = full_batch_counterfactual["experimental"]
    failure_record = full_experimental.get("failure", {})
    placement_record = failure_record.get("placement", {})
    expected_failure = policy["full_batch_failure"]
    if (
        full_experimental["status"] != "failed"
        or failure_record.get("kind") != "placement"
        or failure_record.get("material_id") != expected_failure["material_id"]
        or placement_record.get("status")
        != expected_failure["placement_status"]
    ):
        raise WholeScoreLiveRunError(
            "register exception full batch counterfactual differs: "
            + json.dumps(
                full_batch_counterfactual, ensure_ascii=False, sort_keys=True
            )
        )
    implementation_paths = (
        Path("src/llm_musical_composer/whole_score_staged_generation_run.py"),
        Path("src/llm_musical_composer/staged_material_pilot.py"),
        Path("src/llm_musical_composer/piano_texture_register_placement.py"),
        Path("src/llm_musical_composer/texture_budget.py"),
    )
    return {
        "schema_version": 1,
        "policy_id": policy_id,
        "policy": policy,
        "source_failure": _record_existing_path(project_root, failure_path),
        "source_failure_composition_sha256": sha256_text(source),
        "generation_inputs": {
            "piece_plan": _record_existing_path(project_root, plan_path),
            "harmonic_collection": _record_existing_path(
                project_root, harmonic_collection_path
            ),
            "melody_collection": _record_existing_path(
                project_root, melody_collection_path
            ),
            "texture_budget": _record_existing_path(
                project_root, texture_budget_path
            ),
            "prompt_target_sha256": sha256_json(prompt_target),
        },
        "implementation_hashes": {
            path.as_posix(): sha256_file(project_root / path)
            for path in implementation_paths
        },
        "texture_batch_maximum_event_count": (
            texture_batch_maximum_event_count
        ),
        "first_batch_material_ids": list(first_batch_ids),
        "normal_register_range": normal_range_record,
        "entry_counterfactual": entry_counterfactual,
        "full_batch_counterfactual": full_batch_counterfactual,
    }


def _run_register_enforcement(run_dir: Path) -> dict[str, Any]:
    spec = _read_json(Path(run_dir) / "run-spec.json")
    saved = spec.get("register_enforcement")
    if isinstance(saved, dict):
        requested = saved.get("requested_mode")
        effective = saved.get("effective_mode")
        if requested != effective or requested not in {
            "hard",
            "legacy_prefix_diagnostic_only",
            "diagnostic_only_lower_bound_experiment",
        }:
            raise WholeScoreLiveRunError("run register enforcement is invalid")
        if bool(saved.get("promotion_blocker")) != (
            effective == "diagnostic_only_lower_bound_experiment"
        ):
            raise WholeScoreLiveRunError("run register enforcement is invalid")
        return saved
    target = _read_json(Path(run_dir) / "inputs/prompt-target.json")
    target_version = str(target.get("target_version", ""))
    has_prefix = (Path(run_dir) / "inputs/prepared-texture-prefix.dsl").is_file()
    if has_prefix and _target_version_number(target_version) < 4:
        return {
            "requested_mode": "legacy_prefix_diagnostic_only",
            "effective_mode": "legacy_prefix_diagnostic_only",
            "promotion_blocker": False,
            "source_mode": "persisted-run-spec-compatibility",
        }
    return {
        "requested_mode": "hard",
        "effective_mode": "hard",
        "promotion_blocker": False,
        "source_mode": "persisted-run-spec-default",
    }


def execute_texture_stage(
    project_root: Path,
    run_dir: Path,
    runner: WholeScoreStageRunner,
    skeleton: HarmonicSkeletonV0,
    melodies: ScorePayloadV0,
) -> WholeScoreTextureResultV0:
    """全素材の伴奏集合を生成し、配置可能条件と低域間隔を検査する。"""

    project_root = Path(project_root).resolve()
    run_dir = Path(run_dir).resolve()
    plan = parse_piece_plan(
        (run_dir / "inputs/piece-plan.dsl").read_text(encoding="utf-8")
    )
    prompt_target = _read_json(run_dir / "inputs/prompt-target.json")
    context = build_whole_score_context(plan)
    payloads = {item.material_id: item for item in melodies.materials}
    register_target = _semantic_target_view(prompt_target, "register_envelope")[0]
    register_range_path = run_dir / "outputs/register-range.json"
    if register_range_path.is_file():
        register_range = _read_json(register_range_path)
    else:
        main_ids, _ = melody_generation_order(context)
        register_range = build_allowed_pitch_range(
            plan,
            {
                material_id: [note.pitch for note in payloads[material_id].notes]
                for material_id in main_ids
            },
            register_target,
        )
        atomic_write_json(register_range_path, register_range)
    validate_material_pitch_range({}, register_range)
    allowed_pitch_range = (
        int(register_range["minimum_pitch"]),
        int(register_range["maximum_pitch"]),
    )
    register_enforcement = _run_register_enforcement(run_dir)
    effective_register_mode = register_enforcement["effective_mode"]
    placement_policy = _run_texture_placement_policy(run_dir)
    enforce_register_range = effective_register_mode != (
        "legacy_prefix_diagnostic_only"
    )
    if effective_register_mode == "diagnostic_only_lower_bound_experiment":
        try:
            normal_record = tuple(
                int(item)
                for item in register_enforcement["normal_allowed_pitch_range"]
            )
            experiment_record = tuple(
                int(item)
                for item in register_enforcement[
                    "experimental_allowed_pitch_range"
                ]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise WholeScoreLiveRunError(
                "run register experiment ranges are invalid"
            ) from error
        if (
            normal_record != allowed_pitch_range
            or len(experiment_record) != 2
            or experiment_record[0] != 21
            or experiment_record[1] != allowed_pitch_range[1]
        ):
            raise WholeScoreLiveRunError(
                "run register experiment ranges are invalid"
            )
        placement_allowed_pitch_range = experiment_record
    else:
        placement_allowed_pitch_range = (
            allowed_pitch_range if enforce_register_range else None
        )
    latest_register_progress = measure_register_progress(
        plan,
        {
            material.material_id: [note.pitch for note in material.notes]
            for material in melodies.materials
        },
    )
    ending_requirements = _ending_note_requirements(plan, skeleton)
    release_material_id = context.release_material_ids[0]
    release_melody = payloads[release_material_id]
    release_final_attack = max(
        (note.at_units for note in release_melody.notes), default=None
    )
    ending_assignment_candidates = shared_ending_pitch_assignments(
        foreground_pitches=tuple(
            note.pitch
            for note in release_melody.notes
            if note.at_units == release_final_attack
        ),
        foreground_voice=release_melody.foreground_voice,
        root_pitch_class=plan.tonal_center,
        allowed_pitch_range=allowed_pitch_range,
    )
    if not ending_assignment_candidates:
        raise WholeScoreLiveRunError(
            "shared ending accompaniment is not jointly placeable"
        )
    ending_requirements.update(
        {
            "minimum_required_final_new_accompaniment_attack_count": 2,
            "required_final_pitch_assignment": ending_assignment_candidates[0],
        }
    )
    attack_target = _semantic_target_view(prompt_target, "attack_texture")[0]
    attack_frequency = _run_attack_frequency(run_dir)
    texture_budget = allocate_texture_budget(
        plan,
        skeleton,
        melodies,
        attack_target,
        minimum_attack_group_count=attack_frequency["minimum_attack_group_count"],
    )
    prepared_budget_path = run_dir / "inputs/prepared-texture-budget.json"
    if prepared_budget_path.is_file():
        prepared_budget = _read_json(prepared_budget_path)
        if prepared_budget != texture_budget:
            raise WholeScoreLiveRunError(
                "recomputed texture budget differs from prepared budget"
            )
    atomic_write_json(run_dir / "outputs/texture-budget.json", texture_budget)
    budget_materials = {
        item["material_id"]: item for item in texture_budget["materials"]
    }
    material_order = []
    onset_capacity_reachability = []
    for index, material in enumerate(skeleton.materials, 1):
        melody = payloads[material.material_id]
        material_budget = budget_materials[material.material_id]
        feasibility = build_texture_feasibility(
            material.harmonies,
            melody.notes,
            allowed_pitch_range=placement_allowed_pitch_range,
        )
        start_policy = feasibility["harmony_start_accompaniment"]
        minimum_new_accompaniment_attacks = {
            int(at_units): int(minimum)
            for at_units, minimum in start_policy[
                "minimum_new_accompaniment_attacks"
            ].items()
        }
        last_usable_attack_units = None
        if material.material_id in context.release_material_ids:
            final_attack_units = int(ending_requirements["required_final_attack_units"])
            minimum_new_accompaniment_attacks[final_attack_units] = max(
                minimum_new_accompaniment_attacks.get(final_attack_units, 0), 2
            )
            last_usable_attack_units = final_attack_units
        reachability = texture_budget_onset_capacity_reachability(
            material.length_units,
            melody.notes,
            material_budget,
            feasibility,
            minimum_new_accompaniment_attacks=minimum_new_accompaniment_attacks,
            last_usable_attack_units=last_usable_attack_units,
        )
        onset_capacity_reachability.append(
            {"material_id": material.material_id, **reachability}
        )
        item = {
            "material_id": material.material_id,
            "material_index": index,
            "occurrence_count": material_budget["occurrence_count"],
            "length_units": material.length_units,
            "harmonies": [asdict(item) for item in material.harmonies],
            "foreground_voice": melody.foreground_voice,
            "confirmed_melody": [asdict(item) for item in melody.notes],
            "minimum_event_count": material_budget["minimum_texture_event_count"],
            "maximum_event_count": material_budget["maximum_texture_event_count"],
            "texture_budget": {
                "combined_attack_group_count": material_budget[
                    "combined_attack_group_count"
                ],
                "combined_note_event_count": material_budget[
                    "combined_note_event_count"
                ],
                "attack_size_counts": material_budget["attack_size_counts"],
                "required_texture_event_count": material_budget[
                    "required_texture_event_count"
                ],
            },
            "texture_feasibility": feasibility,
            "minimum_new_accompaniment_attacks": {
                str(at_units): minimum
                for at_units, minimum in sorted(
                    minimum_new_accompaniment_attacks.items()
                )
            },
            "is_transition": material.material_id in context.transition_material_ids,
            "is_release": material.material_id in context.release_material_ids,
        }
        if item["is_release"]:
            item.update(ending_requirements)
        material_order.append(item)
    onset_reachability_path = (
        run_dir / "outputs/texture-onset-capacity-reachability.json"
    )
    atomic_write_json(
        onset_reachability_path,
        {
            "schema_version": 2,
            "materials": onset_capacity_reachability,
            "reachable": all(
                item["reachable"] for item in onset_capacity_reachability
            ),
        },
    )
    if any(not item["reachable"] for item in onset_capacity_reachability):
        raise WholeScoreLiveRunError(
            "texture budget is unreachable under onset capacities"
        )
    run_spec = _read_json(run_dir / "run-spec.json")
    batch_ceiling_value = run_spec.get("texture_batch_maximum_event_count")
    if batch_ceiling_value is not None:
        try:
            batch_ceiling = int(batch_ceiling_value)
        except (TypeError, ValueError) as error:
            raise WholeScoreLiveRunError(
                "texture batch event ceiling is invalid"
            ) from error
        if batch_ceiling <= 0:
            raise WholeScoreLiveRunError("texture batch event ceiling is invalid")
    else:
        batch_ceiling = None
    budget_batches = partition_texture_batches(
        texture_budget["materials"],
        maximum_event_count=batch_ceiling,
    )
    batch_count_ceiling_value = run_spec.get("texture_batch_maximum_count")
    if batch_count_ceiling_value is not None:
        try:
            batch_count_ceiling = int(batch_count_ceiling_value)
        except (TypeError, ValueError) as error:
            raise WholeScoreLiveRunError(
                "texture batch count ceiling is invalid"
            ) from error
        if batch_count_ceiling <= 0:
            raise WholeScoreLiveRunError("texture batch count ceiling is invalid")
        if len(budget_batches) > batch_count_ceiling:
            raise WholeScoreLiveRunError(
                "texture batch count exceeds prepared ceiling"
            )
    material_contexts = {
        item["material_id"]: item for item in material_order
    }
    template = (project_root / _PROMPTS[2]).read_text(encoding="utf-8")
    melody_hash = sha256_json([asdict(item) for item in melodies.materials])
    drafts_by_material: dict[str, Any] = {}
    batch_records = []
    prepared_prefix_path = run_dir / "inputs/prepared-texture-prefix.dsl"
    prepared_batch_count = 0
    if prepared_prefix_path.is_file():
        if batch_ceiling is None:
            raise WholeScoreLiveRunError(
                "prepared texture prefix requires explicit texture batches"
            )
        prepared_source = prepared_prefix_path.read_text(encoding="utf-8")
        prepared_drafts = parse_texture_collection(prepared_source)
        canonical_prefix = dump_texture_collection(prepared_drafts)
        if canonical_prefix != prepared_source:
            raise WholeScoreLiveRunError("prepared texture prefix is not canonical")
        if not 0 < len(prepared_drafts) < len(context.material_ids):
            raise WholeScoreLiveRunError("prepared texture prefix count is invalid")
        prefix_count = len(prepared_drafts)
        consumed = 0
        for budget_batch in budget_batches:
            next_consumed = consumed + len(budget_batch)
            if next_consumed <= prefix_count:
                prepared_batch_count += 1
                consumed = next_consumed
                continue
            if consumed < prefix_count:
                raise WholeScoreLiveRunError(
                    "prepared texture prefix splits a fixed batch"
                )
            break
        if consumed != prefix_count:
            raise WholeScoreLiveRunError(
                "prepared texture prefix does not end at a batch boundary"
            )
        prefix_material_ids = tuple(context.material_ids[:prefix_count])
        drafts_by_material.update(
            zip(prefix_material_ids, prepared_drafts, strict=True)
        )
        prefix_sha256 = sha256_text(canonical_prefix)
        offset = 0
        for batch_index, budget_batch in enumerate(
            budget_batches[:prepared_batch_count], 1
        ):
            batch_material_ids = tuple(
                item["material_id"] for item in budget_batch
            )
            batch_drafts = prepared_drafts[offset : offset + len(budget_batch)]
            offset += len(budget_batch)
            validation = _validate_partial_texture_batches(
                plan,
                skeleton,
                melodies,
                texture_budget,
                {
                    material_id: drafts_by_material[material_id]
                    for material_id in context.material_ids[:offset]
                },
                placement_allowed_pitch_range,
                placement_policy,
            )
            latest_register_progress = validation["register_progress"]
            step_id = f"texture-collection-batch-{batch_index:03d}"
            canonical_batch = dump_texture_collection(batch_drafts)
            _save_text(run_dir / f"outputs/{step_id}.dsl", canonical_batch)
            validation_path = run_dir / f"outputs/{step_id}-validation.json"
            atomic_write_json(validation_path, validation)
            batch_outputs = {
                "texture_draft_sha256": sha256_text(canonical_batch),
                "prepared_texture_prefix_sha256": prefix_sha256,
                "validation_sha256": sha256_file(validation_path),
                "material_ids": list(batch_material_ids),
                "material_count": len(batch_drafts),
                "required_texture_event_count": sum(
                    int(item["required_texture_event_count"])
                    for item in budget_batch
                ),
                "source_mode": "prepared",
                "external_call_number": None,
            }
            _run_store(run_dir).record_step(
                step_id,
                "completed",
                {
                    "harmonic_skeleton": sha256_text(
                        dump_harmonic_skeleton(skeleton)
                    ),
                    "melodies": melody_hash,
                    "prompt_target": sha256_json(prompt_target),
                    "texture_budget": sha256_json(texture_budget),
                    "batch_material_ids": sha256_json(list(batch_material_ids)),
                    "prepared_texture_prefix": prefix_sha256,
                },
                batch_outputs,
            )
            batch_records.append(batch_outputs)
    for batch_index, budget_batch in enumerate(
        budget_batches[prepared_batch_count:], prepared_batch_count + 1
    ):
        batch_material_ids = tuple(item["material_id"] for item in budget_batch)
        step_id = (
            "texture-collection"
            if batch_ceiling is None
            else f"texture-collection-batch-{batch_index:03d}"
        )
        stage_context = {
            "schema_version": 1,
            "piece": {
                "tonal_center": plan.tonal_center,
                "mode": plan.mode,
                "divisions": skeleton.divisions,
            },
            "batch": {
                "index": batch_index,
                "count": len(budget_batches),
                "required_texture_event_count": sum(
                    int(item["required_texture_event_count"])
                    for item in budget_batch
                ),
                "material_ids": list(batch_material_ids),
            },
            "material_order": [
                material_contexts[material_id]
                for material_id in batch_material_ids
            ],
            "controls": prompt_target["controls"],
            "texture_budget": texture_budget["score_spec_target"],
            "semantic_targets": _semantic_target_view(
                prompt_target,
                "attack_texture",
                "foreground_accompaniment_coordination",
                "key_held_texture",
                *(["register_envelope"] if enforce_register_range else []),
            ),
            "attack_frequency": {
                "target_groups_per_second": attack_frequency["target"],
                "candidate_range": [
                    attack_frequency["candidate_minimum"],
                    attack_frequency["candidate_maximum"],
                ],
                "group_tolerance_ms": attack_frequency["group_tolerance_ms"],
            },
            "register_progress": latest_register_progress,
            "register_enforcement": {
                **register_enforcement,
                "normal_allowed_pitch_range": {
                    "minimum_pitch": allowed_pitch_range[0],
                    "maximum_pitch": allowed_pitch_range[1],
                },
                "placement_allowed_pitch_range": (
                    {
                        "minimum_pitch": placement_allowed_pitch_range[0],
                        "maximum_pitch": placement_allowed_pitch_range[1],
                    }
                    if placement_allowed_pitch_range is not None
                    else None
                ),
                "register_span_goal_role": (
                    "diagnostic_only"
                    if effective_register_mode
                    == "diagnostic_only_lower_bound_experiment"
                    else "hard_constraint"
                ),
                "placement_boundary_is_usage_goal": False,
            },
        }
        creative_targets = creative_targets_for_step(prompt_target, step_id)
        if creative_targets:
            stage_context["creative_targets"] = creative_targets
        if enforce_register_range:
            stage_context["allowed_pitch_range"] = {
                "minimum_pitch": placement_allowed_pitch_range[0],
                "maximum_pitch": placement_allowed_pitch_range[1],
            }
        prompt = _render_prompt(template, stage_context)
        _save_text(run_dir / f"prompts/{step_id}.md", prompt)
        input_hashes = {
            "harmonic_skeleton": sha256_text(dump_harmonic_skeleton(skeleton)),
            "melodies": melody_hash,
            "prompt_target": sha256_json(prompt_target),
            "texture_budget": sha256_json(texture_budget),
            "batch_material_ids": sha256_json(list(batch_material_ids)),
            "prompt": sha256_text(prompt),
            **_creative_target_input_hashes(prompt_target, step_id),
        }
        response = runner.run(step_id, prompt, input_hashes)
        source = _response_source(response, step_id)
        _save_text(run_dir / f"responses/{step_id}.dsl", source)
        try:
            batch_drafts = parse_texture_collection(source)
            if len(batch_drafts) != len(batch_material_ids):
                raise WholeScoreLiveRunError(
                    f"{step_id} count does not match fixed order"
                )
            drafts_by_material.update(
                zip(batch_material_ids, batch_drafts, strict=True)
            )
            validation = _validate_partial_texture_batches(
                plan,
                skeleton,
                melodies,
                texture_budget,
                drafts_by_material,
                placement_allowed_pitch_range,
                placement_policy,
            )
        except Exception as error:
            _save_stage_failure(run_dir, step_id, source, error, input_hashes)
            raise
        latest_register_progress = validation["register_progress"]
        canonical_batch = dump_texture_collection(batch_drafts)
        _save_text(run_dir / f"outputs/{step_id}.dsl", canonical_batch)
        validation_path = run_dir / f"outputs/{step_id}-validation.json"
        atomic_write_json(validation_path, validation)
        batch_outputs = {
            "texture_draft_sha256": sha256_text(canonical_batch),
            "validation_sha256": sha256_file(validation_path),
            "material_ids": list(batch_material_ids),
            "material_count": len(batch_drafts),
            "required_texture_event_count": stage_context["batch"][
                "required_texture_event_count"
            ],
            "source_mode": "external",
            "external_call_number": runner.call_number,
        }
        if batch_ceiling is not None:
            _run_store(run_dir).record_step(
                step_id,
                "completed",
                input_hashes,
                batch_outputs,
            )
        batch_records.append(batch_outputs)
    drafts = tuple(
        drafts_by_material[material_id] for material_id in context.material_ids
    )
    result = assemble_whole_score_textures(
        "reference-v7-a2-live-v1",
        plan,
        skeleton,
        melodies,
        drafts,
        maximum_event_counts={
            material_id: item["maximum_texture_event_count"]
            for material_id, item in budget_materials.items()
        },
        placement_policy=placement_policy,
        allowed_pitch_range=placement_allowed_pitch_range,
    )
    register_diagnostic = measure_register_diagnostic(
        plan,
        result.score,
        register_target,
    )
    register_diagnostic["register_enforcement"] = register_enforcement
    register_diagnostic["normal_allowed_pitch_range"] = {
        "minimum_pitch": allowed_pitch_range[0],
        "maximum_pitch": allowed_pitch_range[1],
    }
    register_diagnostic["placement_allowed_pitch_range"] = (
        {
            "minimum_pitch": placement_allowed_pitch_range[0],
            "maximum_pitch": placement_allowed_pitch_range[1],
        }
        if placement_allowed_pitch_range is not None
        else None
    )
    atomic_write_json(
        run_dir / "outputs/register-diagnostic.json",
        register_diagnostic,
    )
    score_budget = measure_score_texture_budget(plan, result.score, texture_budget)
    atomic_write_json(run_dir / "outputs/score-texture-budget.json", score_budget)
    if not score_budget["whole_score"]["matches_budget"]:
        error = WholeScoreLiveRunError(
            "generated score does not match texture budget"
        )
        _save_stage_failure(
            run_dir,
            "texture-collection",
            source,
            error,
            input_hashes,
        )
        raise error
    canonical = dump_texture_collection(drafts)
    score_source = dump_score_spec(result.score)
    _save_text(run_dir / "outputs/texture-collection.dsl", canonical)
    _save_text(run_dir / "outputs/score-spec.dsl", score_source)
    atomic_write_json(
        run_dir / "outputs/texture-placement.json",
        {
            "schema_version": (
                5
                if placement_policy == "search-aware-onset-zone-v8"
                else 4
                if placement_policy == "onset-feasible-zone-v7"
                else 3
                if placement_policy == "bidirectional-low-spacing-v6"
                else 2
            ),
            "placement_policy": placement_policy,
            "placements": [asdict(item) for item in result.placements],
            "low_spacing_violations": list(result.low_spacing_violations),
        },
    )
    if batch_ceiling is not None:
        _run_store(run_dir).record_step(
            "texture-collection",
            "completed",
            {
                "harmonic_skeleton": sha256_text(dump_harmonic_skeleton(skeleton)),
                "melodies": melody_hash,
                "prompt_target": sha256_json(prompt_target),
                "texture_budget": sha256_json(texture_budget),
                "batch_outputs": sha256_json(batch_records),
            },
            {
                "texture_draft_sha256": sha256_text(canonical),
                "score_spec_sha256": sha256_text(score_source),
                "texture_budget_sha256": sha256_json(texture_budget),
                "score_texture_budget_sha256": sha256_json(score_budget),
                "material_count": len(drafts),
                "batch_count": len(batch_records),
                "source_mode": "external-batches",
                "external_call_number": None,
            },
        )
    else:
        _run_store(run_dir).record_step(
            "texture-collection",
            "completed",
            input_hashes,
            {
                "texture_draft_sha256": sha256_text(canonical),
                "score_spec_sha256": sha256_text(score_source),
                "texture_budget_sha256": sha256_json(texture_budget),
                "score_texture_budget_sha256": sha256_json(score_budget),
                "material_count": len(drafts),
                "external_call_number": runner.call_number,
            },
        )
    return result


def execute_performance_stage(
    project_root: Path,
    run_dir: Path,
    runner: WholeScoreStageRunner,
    score: ScoreSpec,
) -> LivePerformanceStageResult:
    """全出現の演奏指定を生成し、180秒の候補をstaged領域へ描画する。"""

    project_root = Path(project_root).resolve()
    run_dir = Path(run_dir).resolve()
    plan = parse_piece_plan(
        (run_dir / "inputs/piece-plan.dsl").read_text(encoding="utf-8")
    )
    prompt_target = _read_json(run_dir / "inputs/prompt-target.json")
    texture_budget = _read_json(run_dir / "outputs/texture-budget.json")
    leaves = _ordered_leaves(plan)
    materials = {item.material_id: item for item in score.materials}
    occurrence_numbers: dict[str, int] = {}
    occurrences = []
    for position, leaf in enumerate(leaves, 1):
        assert leaf.score_material_id is not None
        material_id = leaf.score_material_id
        occurrence_numbers[material_id] = occurrence_numbers.get(material_id, 0) + 1
        material = materials[material_id]
        occurrences.append(
            {
                "position": position,
                "role": leaf.role,
                "material_index": tuple(materials).index(material_id) + 1,
                "material_occurrence_number": occurrence_numbers[material_id],
                "is_repeated_material": sum(
                    item.score_material_id == material_id for item in leaves
                )
                > 1,
                "note_count": len(material.notes),
                "foreground_voice": material.foreground_voice,
                "harmony_count": len(material.harmonies),
                "is_release": leaf.role == "release",
            }
        )
    context = {
        "schema_version": 1,
        "piece": {
            "target_duration_ms": WHOLE_SCORE_TARGET_DURATION_MS,
            "occurrence_count": len(leaves),
        },
        "occurrence_order": occurrences,
        "controls": prompt_target["controls"],
        "texture_budget": texture_budget["score_spec_target"],
        "semantic_targets": _semantic_target_view(
            prompt_target,
            "coordination_preservation",
            "key_held_texture",
            "velocity_shape",
        ),
        "allowed_vocabulary": {
            "timing_profile": ["neutral", "savor", "flow", "build", "release", None],
            "timing_amount": ["subtle", "moderate", None],
            "dynamics_profile": ["steady", "shape", "build", "release", None],
            "articulation_profile": ["score", "legato", "light", None],
            "coordination_profile": ["score", "rolled", "aligned", None],
            "pedal_profile": [
                "none",
                "phrase_legato",
                "harmony_legato",
                None,
            ],
        },
    }
    creative_targets = creative_targets_for_step(
        prompt_target, "performance-collection"
    )
    if creative_targets:
        context["creative_targets"] = creative_targets
    template = (project_root / _PROMPTS[3]).read_text(encoding="utf-8")
    prompt = _render_prompt(template, context)
    _save_text(run_dir / "prompts/performance-collection.md", prompt)
    score_source = dump_score_spec(score)
    input_hashes = {
        "score_spec": sha256_text(score_source),
        "prompt_target": sha256_json(prompt_target),
        "texture_budget": sha256_json(texture_budget),
        "prompt": sha256_text(prompt),
        **_creative_target_input_hashes(prompt_target, "performance-collection"),
    }
    response = runner.run(
        "performance-collection",
        prompt,
        input_hashes,
    )
    source = _response_source(response, "performance collection")
    _save_text(run_dir / "responses/performance-collection.dsl", source)
    drafts = parse_performance_collection(source)
    if len(drafts) != len(leaves):
        raise WholeScoreLiveRunError("performance collection count does not match fixed order")
    performance = assemble_whole_score_performance(
        "reference-v7-a2-live-v1",
        plan,
        score,
        drafts,
    )
    run_spec = _read_json(run_dir / "run-spec.json")
    performance = replace(
        performance,
        velocity_policy_id=str(
            run_spec.get("velocity_policy_id", "legacy-unison-v1")
        ),
    )
    velocity_target = _semantic_target_view(prompt_target, "velocity_shape")[0]
    key_release_target = _semantic_target_view(prompt_target, "key_held_texture")[0]
    try:
        performance, rendered, velocity_calibration = calibrate_default_velocity(
            plan,
            score,
            performance,
            velocity_target,
        )
        performance, rendered, key_release_calibration = calibrate_key_release(
            plan,
            score,
            performance,
            key_release_target,
            on_unreachable=str(
                run_spec.get("key_release_unreachable_policy", "raise")
            ),
        )
    except WholeScoreLiveRunError as error:
        _save_stage_failure(
            run_dir,
            "performance-collection",
            source,
            error,
            input_hashes,
        )
        raise
    atomic_write_json(
        run_dir / "outputs/velocity-calibration.json",
        velocity_calibration,
    )
    atomic_write_json(
        run_dir / "outputs/key-release-calibration.json",
        key_release_calibration,
    )
    voice_velocity = analyze_voice_velocity(plan, score, performance)
    atomic_write_json(
        run_dir / "outputs/voice-velocity-diagnostic.json",
        voice_velocity,
    )
    rendered_budget = measure_rendered_texture_budget(rendered, texture_budget)
    atomic_write_json(
        run_dir / "outputs/rendered-texture-budget.json",
        rendered_budget,
    )
    rendered_measurement = rendered_budget["rendered_performance"]
    frequency_contract = _run_attack_frequency(run_dir)
    rendered_passes = _rendered_texture_budget_passes(
        frequency_contract,
        rendered_measurement,
    )
    if not rendered_passes:
        error = WholeScoreLiveRunError(
            "rendered performance does not match texture budget"
        )
        _save_stage_failure(
            run_dir,
            "performance-collection",
            source,
            error,
            input_hashes,
        )
        raise error
    score_quality = evaluate_generic_score_quality(plan, score)
    pipeline_quality = evaluate_generic_pipeline_quality(
        plan,
        score,
        performance,
        rendered,
    )
    atomic_write_json(
        run_dir / "outputs/generic-quality.json",
        {"score": score_quality, "pipeline": pipeline_quality},
    )
    if not score_quality.get("passes") or not pipeline_quality.get("passes"):
        raise WholeScoreLiveRunError(
            "generated whole score failed generic quality: "
            f"score={score_quality.get('failures')}, "
            f"pipeline={pipeline_quality.get('failures')}"
        )
    canonical = dump_performance_collection(drafts)
    performance_source = dump_performance_spec(performance)
    _save_text(run_dir / "outputs/performance-collection.dsl", canonical)
    _save_text(run_dir / "outputs/performance-spec.dsl", performance_source)
    staged_dir = run_dir / "staged"
    musicxml_path = render_musicxml(plan, score, staged_dir / "final.musicxml")
    smf_path = render_performance_smf(rendered, staged_dir / "final.mid").path
    _run_store(run_dir).record_step(
        "performance-collection",
        "completed",
        input_hashes,
        {
            "performance_spec_sha256": sha256_text(performance_source),
            "musicxml_sha256": sha256_file(musicxml_path),
            "smf_sha256": sha256_file(smf_path),
            "generic_quality": {
                "score_passes": score_quality["passes"],
                "pipeline_passes": pipeline_quality["passes"],
            },
            "rendered_texture_budget_sha256": sha256_json(rendered_budget),
            "key_release_calibration_sha256": sha256_json(key_release_calibration),
            "velocity_calibration_sha256": sha256_json(velocity_calibration),
            "voice_velocity_diagnostic_sha256": sha256_json(voice_velocity),
            "external_call_number": runner.call_number,
        },
    )
    return LivePerformanceStageResult(
        performance,
        rendered,
        score_quality,
        pipeline_quality,
        musicxml_path,
        smf_path,
    )


def _metric_distance(kind: str, actual: list[float], expected: list[float]) -> float:
    if len(actual) != len(expected):
        raise WholeScoreLiveRunError("reference descriptor dimensions differ")
    if kind == "scalar":
        if len(actual) != 1:
            raise WholeScoreLiveRunError("scalar descriptor must have one value")
        return abs(actual[0] - expected[0])
    if kind == "distribution":
        return sum(abs(left - right) for left, right in zip(actual, expected, strict=True)) / 2
    raise WholeScoreLiveRunError(f"unknown reference descriptor kind: {kind}")


def _rendered_key_held_distribution(rendered: RenderedPerformance) -> list[float]:
    if not rendered.notes:
        raise WholeScoreLiveRunError("key release calibration needs rendered notes")
    start = min(note.at_ms for note in rendered.notes)
    end = max(note.at_ms + note.duration_ms for note in rendered.notes)
    if end <= start:
        raise WholeScoreLiveRunError("key release calibration duration is invalid")
    deltas: dict[int, int] = {}
    for note in rendered.notes:
        deltas[note.at_ms] = deltas.get(note.at_ms, 0) + 1
        release = note.at_ms + note.duration_ms
        deltas[release] = deltas.get(release, 0) - 1
    durations = [0, 0, 0, 0, 0]
    active = 0
    previous = start
    for at_ms in sorted(deltas):
        if at_ms < start or at_ms > end:
            continue
        if at_ms > previous:
            durations[min(active, 4)] += at_ms - previous
        active += deltas[at_ms]
        if active < 0:
            raise WholeScoreLiveRunError(
                "key release calibration active polyphony became negative"
            )
        previous = at_ms
    if previous < end:
        durations[min(active, 4)] += end - previous
    total = end - start
    return [duration / total for duration in durations]


def _terminal_exempt_note_count(plan: PiecePlan, score: ScoreSpec) -> int:
    final_leaf = _ordered_leaves(plan)[-1]
    assert final_leaf.score_material_id is not None
    material = next(
        item for item in score.materials if item.material_id == final_leaf.score_material_id
    )
    final_onset = max(note.at_units for note in material.notes)
    return sum(
        note.at_units == final_onset
        and note.at_units + note.duration_units == material.length_units
        for note in material.notes
    )


def calibrate_key_release(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
    target: dict[str, Any],
    *,
    minimum_percent: int = 40,
    on_unreachable: str = "raise",
) -> tuple[PerformanceSpec, RenderedPerformance, dict[str, Any]]:
    """キー保持分布へ入る最大の整数解放率を選ぶ。"""

    if not 40 <= minimum_percent <= 100:
        raise WholeScoreLiveRunError("key release calibration range is invalid")
    if on_unreachable not in {"raise", "nearest_unfit"}:
        raise WholeScoreLiveRunError("key release unreachable policy is invalid")
    center_raw = target.get("neighborhood_center")
    radius_raw = target.get("neighborhood_radius")
    if not isinstance(center_raw, list) or len(center_raw) != 5:
        raise WholeScoreLiveRunError("key release calibration target is invalid")
    try:
        center = [float(value) for value in center_raw]
        radius = float(radius_raw)
    except (TypeError, ValueError) as error:
        raise WholeScoreLiveRunError(
            "key release calibration target is invalid"
        ) from error
    if radius < 0 or not math.isfinite(radius) or any(
        value < 0 or not math.isfinite(value) for value in center
    ):
        raise WholeScoreLiveRunError("key release calibration target is invalid")
    base = replace(performance, key_release_percent=100)
    before_rendered = render_performance(plan, score, base)
    before_actual = _rendered_key_held_distribution(before_rendered)
    before_distance = _metric_distance("distribution", before_actual, center)
    evaluated = []
    for percent in range(100, minimum_percent - 1, -1):
        candidate = replace(base, key_release_percent=percent)
        rendered = render_performance(plan, score, candidate)
        actual = _rendered_key_held_distribution(rendered)
        distance = _metric_distance("distribution", actual, center)
        evaluated.append((distance, percent, candidate, rendered, actual))
        if distance <= radius:
            return candidate, rendered, {
                "schema_version": 1,
                "status": "pass",
                "policy_id": "largest-passing-integer-percent-v1",
                "rounding_policy": "single-round-after-articulation-and-percent-v1",
                "search": {
                    "maximum_percent": 100,
                    "minimum_percent": minimum_percent,
                    "step_percent": 1,
                },
                "selected_percent": percent,
                "terminal_exempt_note_count": _terminal_exempt_note_count(plan, score),
                "before": {
                    "actual": before_actual,
                    "distance": before_distance,
                },
                "after": {
                    "actual": actual,
                    "distance": distance,
                },
                "target": {
                    "neighborhood_center": center,
                    "neighborhood_radius": radius,
                },
            }
    if on_unreachable == "nearest_unfit":
        distance, percent, candidate, rendered, actual = min(
            evaluated,
            key=lambda item: (item[0], -item[1]),
        )
        return candidate, rendered, {
            "schema_version": 1,
            "status": "nearest_unfit",
            "passes": False,
            "policy_id": "nearest-unfit-integer-percent-v1",
            "rounding_policy": "single-round-after-articulation-and-percent-v1",
            "search": {
                "maximum_percent": 100,
                "minimum_percent": minimum_percent,
                "step_percent": 1,
            },
            "selected_percent": percent,
            "terminal_exempt_note_count": _terminal_exempt_note_count(plan, score),
            "before": {
                "actual": before_actual,
                "distance": before_distance,
            },
            "after": {
                "actual": actual,
                "distance": distance,
            },
            "target": {
                "neighborhood_center": center,
                "neighborhood_radius": radius,
            },
            "radius_excess": distance - radius,
        }
    raise WholeScoreLiveRunError(
        "key release calibration has no passing integer percent"
    )


def _velocity_distribution(rendered: RenderedPerformance) -> list[float]:
    counts = [0] * 8
    for note in rendered.notes:
        counts[min(note.velocity // 16, 7)] += 1
    total = len(rendered.notes)
    if total == 0:
        raise WholeScoreLiveRunError("velocity calibration has no notes")
    return [count / total for count in counts]


def _velocity_clip_count(
    plan: PiecePlan, score: ScoreSpec, performance: PerformanceSpec
) -> int:
    materials = {item.material_id: item for item in score.materials}
    nodes = {item.node_id: item for item in plan.nodes}
    profiles = {item.node_id: item for item in performance.node_performances}
    count = 0
    for leaf in _ordered_leaves(plan):
        assert leaf.score_material_id is not None
        material = materials[leaf.score_material_id]
        _, profile = resolve_effective_profile(
            leaf, nodes, profiles, "dynamics_profile", "steady"
        )
        adjustments = material_velocity_adjustments(
            material, performance.velocity_policy_id
        )
        count += sum(
            not 1
            <= dynamic_velocity_unclamped(
                material, note, performance.default_velocity, profile
            )
            + adjustments[note.event_id]
            <= 127
            for note in material.notes
        )
    return count


def calibrate_default_velocity(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
    target: dict[str, Any],
) -> tuple[PerformanceSpec, RenderedPerformance, dict[str, Any]]:
    """profileを固定し、参照velocity分布へ入るclipなしの既定値を選ぶ。"""

    try:
        center = [float(value) for value in target["neighborhood_center"]]
        radius = float(target["neighborhood_radius"])
    except (KeyError, TypeError, ValueError) as error:
        raise WholeScoreLiveRunError("velocity calibration target is invalid") from error
    if len(center) != 8 or radius < 0:
        raise WholeScoreLiveRunError("velocity calibration target is invalid")
    before = render_performance(plan, score, performance)
    before_distribution = _velocity_distribution(before)
    candidates = []
    for value in range(1, 128):
        candidate = replace(performance, default_velocity=value)
        if _velocity_clip_count(plan, score, candidate):
            continue
        rendered = render_performance(plan, score, candidate)
        distribution = _velocity_distribution(rendered)
        distance = _metric_distance("distribution", distribution, center)
        if distance <= radius:
            candidates.append(
                (
                    distance,
                    abs(value - performance.default_velocity),
                    value,
                    candidate,
                    rendered,
                    distribution,
                )
            )
    if not candidates:
        raise WholeScoreLiveRunError("velocity calibration has no passing unclipped integer")
    distance, _, value, candidate, rendered, distribution = min(
        candidates, key=lambda item: item[:3]
    )

    def summary(
        result: RenderedPerformance,
        specification: PerformanceSpec,
    ) -> dict[str, Any]:
        values = [note.velocity for note in result.notes]
        return {
            "distribution": _velocity_distribution(result),
            "distinct_velocity_count": len(set(values)),
            "minimum": min(values),
            "maximum": max(values),
            "clip_count": _velocity_clip_count(plan, score, specification),
        }
    return candidate, rendered, {
        "schema_version": 1,
        "status": "pass",
        "policy_id": "nearest-unclipped-reference-velocity-v1",
        "selected_default_velocity": value,
        "clip_count": 0,
        "before": {
            **summary(before, performance),
            "distance": _metric_distance(
                "distribution", before_distribution, center
            ),
        },
        "after": {**summary(rendered, candidate), "distance": distance},
        "target": {"neighborhood_center": center, "neighborhood_radius": radius},
    }


def _measure_descriptors(
    prompt_target: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    targets = {
        item["id"]: (stage, item)
        for stage, items in prompt_target["stage_targets"].items()
        for item in items
    }
    expected_ids = set(_descriptor_ids(prompt_target))
    records = []
    observed_ids: set[str] = set()
    for descriptor_id, (generation_stage, target) in targets.items():
        group, metric_name = descriptor_id.split(".", 1)
        try:
            metric = profile["feature_groups"][group]["metrics"][metric_name]
        except (KeyError, TypeError) as error:
            raise WholeScoreLiveRunError(
                f"generated profile lacks descriptor: {descriptor_id}"
            ) from error
        kind = str(metric["kind"])
        actual = [float(value) for value in metric["values"]]
        if kind != target["kind"]:
            raise WholeScoreLiveRunError(f"descriptor kind differs: {descriptor_id}")
        center = [float(value) for value in target["neighborhood_center"]]
        anchor = [float(value) for value in target["anchor"]]
        center_distance = _metric_distance(kind, actual, center)
        radius = float(target["neighborhood_radius"])
        records.append(
            {
                "id": descriptor_id,
                "kind": kind,
                "generation_stage": generation_stage,
                "final_observation_stage": target["final_observation_stage"],
                "actual": actual,
                "anchor_distance": _metric_distance(kind, actual, anchor),
                "neighborhood_center_distance": center_distance,
                "neighborhood_center_residuals": [
                    value - expected for value, expected in zip(actual, center, strict=True)
                ],
                "neighborhood_radius": radius,
                "within_neighborhood": center_distance <= radius,
            }
        )
        observed_ids.add(descriptor_id)
    if observed_ids != expected_ids:
        raise WholeScoreLiveRunError("measured descriptor IDs differ from prompt target")
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise WholeScoreLiveRunError(f"JSONL row must be an object: {path}")
            records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WholeScoreLiveRunError(f"cannot read JSONL input: {path}") from error
    return records


def _verify_file_records(project_root: Path, records: dict[str, Any]) -> None:
    for name, record in records.items():
        if not isinstance(record, dict):
            raise WholeScoreLiveRunError(f"input record is invalid: {name}")
        path = project_root / str(record.get("path", ""))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise WholeScoreLiveRunError(f"prepared input drifted: {name}")


def _aggregate_overlap_distribution(
    raw: object,
    *,
    minimum_key: int,
    upper_bin: int,
) -> list[float]:
    if not isinstance(raw, dict) or not raw:
        raise WholeScoreLiveRunError("overlap distribution is invalid")
    values = [0.0] * (upper_bin - minimum_key + 1)
    for raw_key, raw_value in raw.items():
        try:
            key = int(raw_key)
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise WholeScoreLiveRunError("overlap distribution entry is invalid") from error
        if key < minimum_key or value < 0:
            raise WholeScoreLiveRunError("overlap distribution entry is invalid")
        values[min(key, upper_bin) - minimum_key] += value
    total = sum(values)
    if total <= 0:
        raise WholeScoreLiveRunError("overlap distribution is empty")
    return [value / total for value in values]


def _measure_semantic_texture(
    prompt_target: dict[str, Any],
    control_observables: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    try:
        overlap = control_observables["diagnostics"]["overlap"]
    except (KeyError, TypeError) as error:
        raise WholeScoreLiveRunError("generated overlap diagnostics are missing") from error
    actual_by_id = {
        "attack_texture": _aggregate_overlap_distribution(
            overlap["attack_size_distribution"],
            minimum_key=1,
            upper_bin=4,
        ),
        "key_held_texture": _aggregate_overlap_distribution(
            overlap["active_polyphony_duration_distribution"],
            minimum_key=0,
            upper_bin=4,
        ),
    }
    records: dict[str, dict[str, Any]] = {}
    for target_id, actual in actual_by_id.items():
        target = _semantic_target_view(prompt_target, target_id)[0]
        center = [float(value) for value in target["neighborhood_center"]]
        radius = float(target["neighborhood_radius"])
        distance = _metric_distance("distribution", actual, center)
        records[target_id] = {
            "actual": actual,
            "neighborhood_center": center,
            "neighborhood_center_distance": distance,
            "neighborhood_radius": radius,
            "within_neighborhood": distance <= radius,
        }
    attack_target = _semantic_target_view(prompt_target, "attack_texture")[0]
    notes_target = attack_target["notes_per_attack"]
    notes_actual = float(overlap["notes_per_attack"])
    notes_center = float(notes_target["neighborhood_center"])
    notes_radius = float(notes_target["neighborhood_radius"])
    raw_attack_sizes = overlap["attack_size_distribution"]
    maximum_actual = max(
        int(key) for key, value in raw_attack_sizes.items() if float(value) > 0
    )
    maximum_limit = int(attack_target["maximum_group_size"]["anchor"])
    records["attack_texture"].update(
        {
            "group_tolerance_ms": 30,
            "notes_per_attack": notes_actual,
            "notes_per_attack_neighborhood_center": notes_center,
            "notes_per_attack_neighborhood_distance": abs(
                notes_actual - notes_center
            ),
            "notes_per_attack_neighborhood_radius": notes_radius,
            "notes_per_attack_within_neighborhood": abs(notes_actual - notes_center)
            <= notes_radius,
            "maximum_group_size": maximum_actual,
            "maximum_group_size_limit": maximum_limit,
            "maximum_group_size_within_limit": maximum_actual <= maximum_limit,
        }
    )
    records["key_held_texture"]["includes_pedal_extension"] = False
    return records


def _voice_alignment(notes: list[Any], tolerance_ms: int) -> float:
    upper = [note.at_ms for note in notes if note.voice == "upper"]
    lower = [note.at_ms for note in notes if note.voice == "lower"]
    if not upper or not lower:
        return 0.0
    matched_upper = sum(
        any(abs(at_ms - other) <= tolerance_ms for other in lower) for at_ms in upper
    )
    matched_lower = sum(
        any(abs(at_ms - other) <= tolerance_ms for other in upper) for at_ms in lower
    )
    return (matched_upper + matched_lower) / (len(upper) + len(lower))


def _measure_texture_alignment(
    plan: PiecePlan,
    score: ScoreSpec,
    rendered: RenderedPerformance,
) -> dict[str, Any]:
    materials = {material.material_id: material for material in score.materials}
    material_records = []
    for material in score.materials:
        assessment = analyze_material_vertical_alignment(material, minimum_ratio=0.0)
        material_records.append(
            {
                "material_id": material.material_id,
                "score_shared_attack_ratio": assessment.shared_attack_ratio,
                "internal_independence": assessment.internal_independence,
            }
        )
    occurrence_records = []
    for leaf in _ordered_leaves(plan):
        assert leaf.score_material_id is not None
        notes = [
            note for note in rendered.notes if note.occurrence_node_id == leaf.node_id
        ]
        score_assessment = analyze_material_vertical_alignment(
            materials[leaf.score_material_id], minimum_ratio=0.0
        )
        occurrence_records.append(
            {
                "node_id": leaf.node_id,
                "material_id": leaf.score_material_id,
                "score_shared_attack_ratio": score_assessment.shared_attack_ratio,
                "performed_alignment_10ms": _voice_alignment(notes, 10),
                "performed_alignment_30ms": _voice_alignment(notes, 30),
            }
        )
    return {
        "material_count": len(material_records),
        "occurrence_count": len(occurrence_records),
        "materials": material_records,
        "occurrences": occurrence_records,
    }


def _semantic_texture_fit(semantic_texture: dict[str, dict[str, Any]]) -> bool:
    attack = semantic_texture["attack_texture"]
    return (
        bool(attack["within_neighborhood"])
        and bool(attack["notes_per_attack_within_neighborhood"])
        and bool(attack["maximum_group_size_within_limit"])
        and bool(semantic_texture["key_held_texture"]["within_neighborhood"])
    )


def _validate_rendered_outputs(
    run_dir: Path,
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
    rendered: RenderedPerformance,
    musicxml_path: Path,
    smf_path: Path,
) -> dict[str, Any]:
    diagnostic = {
        "schema_version": 1,
        "status": "passed",
        "musicxml": validate_musicxml_round_trip(plan, score, musicxml_path),
        "smf": validate_smf_round_trip(rendered, smf_path),
    }
    if any(diagnostic[item]["status"] != "passed" for item in ("musicxml", "smf")):
        diagnostic["status"] = "failed"
    diagnostic_path = run_dir / "outputs/rendered-output-validation.json"
    atomic_write_json(diagnostic_path, diagnostic)
    input_hashes = {
        "piece_plan": sha256_file(run_dir / "inputs/piece-plan.dsl"),
        "score_spec": sha256_file(run_dir / "outputs/score-spec.dsl"),
        "performance_spec": sha256_file(run_dir / "outputs/performance-spec.dsl"),
        "staged_musicxml": sha256_file(musicxml_path),
        "staged_smf": sha256_file(smf_path),
    }
    _run_store(run_dir).record_step(
        "rendered-output-validation",
        "completed" if diagnostic["status"] == "passed" else "failed",
        input_hashes,
        {
            "status": diagnostic["status"],
            "diagnostic_path": str(diagnostic_path.relative_to(run_dir)),
            "diagnostic_sha256": sha256_file(diagnostic_path),
            "musicxml_sha256": diagnostic["musicxml"]["sha256"],
            "smf_sha256": diagnostic["smf"]["sha256"],
        },
    )
    if diagnostic["status"] != "passed":
        raise WholeScoreLiveRunError("rendered output round-trip validation failed")
    return diagnostic


def evaluate_staged_candidate(
    project_root: Path,
    run_dir: Path,
    *,
    candidate_use: str = "external_generation",
) -> dict[str, Any]:
    """staged SMFを参照記述子、3制御、copy、発音頻度で評価して昇格する。"""

    promotion_eligible = candidate_use_allows_promotion(candidate_use)

    project_root = Path(project_root).resolve()
    run_dir = Path(run_dir).resolve()
    spec = _read_json(run_dir / "run-spec.json")
    input_hashes = spec.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise WholeScoreLiveRunError("prepared input hashes are missing")
    _verify_file_records(project_root, input_hashes["evaluation_inputs"])
    smf_path = run_dir / "staged/final.mid"
    musicxml_path = run_dir / "staged/final.musicxml"
    if not smf_path.is_file() or not musicxml_path.is_file():
        raise WholeScoreLiveRunError("staged rendered outputs are missing")
    plan = parse_piece_plan((run_dir / "inputs/piece-plan.dsl").read_text(encoding="utf-8"))
    score = parse_score_spec((run_dir / "outputs/score-spec.dsl").read_text(encoding="utf-8"))
    performance = parse_performance_spec(
        (run_dir / "outputs/performance-spec.dsl").read_text(encoding="utf-8")
    )
    rendered = render_performance(plan, score, performance)
    rendered_output_validation = _validate_rendered_outputs(
        run_dir,
        plan,
        score,
        performance,
        rendered,
        musicxml_path,
        smf_path,
    )
    prompt_target = _read_json(run_dir / "inputs/prompt-target.json")
    piece = load_reference_piece(smf_path)
    profile = extract_reference_profile(piece)
    descriptors = _measure_descriptors(prompt_target, profile)
    controls = extract_control_observables(piece)
    semantic_texture = _measure_semantic_texture(prompt_target, controls)
    texture_alignment = _measure_texture_alignment(plan, score, rendered)
    raw = controls["raw"]
    frequency_contract = _run_attack_frequency(run_dir)
    control_summary = _read_json(
        project_root / ".appendix/control-reference-baseline-v3/summary.json"
    )
    labels = {
        "brightness": "あかるさ",
        "height": "高さ",
        "attack_frequency": "発音頻度",
    }
    measured_controls = {}
    for control_id, label in labels.items():
        axis = control_summary["axes"][label]
        value = float(raw[label])
        normalized = normalize_value(
            value,
            minimum=float(axis["minimum"]),
            maximum=float(axis["maximum"]),
        )
        measured_controls[control_id] = {
            "raw": value,
            "normalized": normalized,
            "target_normalized": float(prompt_target["controls"][control_id]),
            "normalized_residual": normalized
            - float(prompt_target["controls"][control_id]),
        }
    tonal_targets = _specified_tonal_targets(prompt_target)
    brightness_resolution = evaluate_brightness_resolution(
        measured_controls["brightness"]["normalized"],
        tonal_targets[0] if tonal_targets else {"status": "unverified_continuous_reference"},
    )
    measured_controls["brightness"]["resolution"] = brightness_resolution
    attack_frequency = float(raw["発音頻度"])
    rendered_attack_group_count = int(
        controls["diagnostics"]["attack_group_count"]
    )
    target_attack_group_count = int(
        frequency_contract["minimum_attack_group_count"]
    )
    within_frequency = (
        rendered_attack_group_count == target_attack_group_count
        if frequency_contract["source"] == "explicit_corpus_min_max_v1"
        else frequency_contract["candidate_minimum"]
        <= attack_frequency
        <= frequency_contract["candidate_maximum"]
    )
    score_texture_budget_path = run_dir / "outputs/score-texture-budget.json"
    if score_texture_budget_path.is_file():
        score_attack_group_count = int(
            _read_json(score_texture_budget_path)["whole_score"][
                "attack_group_count"
            ]
        )
    elif frequency_contract["source"] == "explicit_corpus_min_max_v1":
        raise WholeScoreLiveRunError(
            "explicit attack frequency score diagnostic is missing"
        )
    else:
        score_attack_group_count = None

    evaluation_records = input_hashes["evaluation_inputs"]
    profile_record = evaluation_records["reference_profiles"]
    references = {
        str(item["name"]): item["copy_fingerprint"]
        for item in _read_jsonl(project_root / str(profile_record["path"]))
        if isinstance(item.get("name"), str)
        and isinstance(item.get("copy_fingerprint"), dict)
    }
    for name in ("capability_v26_smf", "multiscale_v8_smf"):
        path = project_root / str(evaluation_records[name]["path"])
        references[f"evaluation:{name}"] = build_copy_fingerprint(load_reference_piece(path))
    if "source_v7_a2_smf" in evaluation_records:
        path = project_root / str(evaluation_records["source_v7_a2_smf"]["path"])
        references["evaluation:source_v7_a2_smf"] = build_copy_fingerprint(
            load_reference_piece(path)
        )
    copy_risk = evaluate_copy_risk(
        build_copy_fingerprint(piece),
        references,
        review_threshold=0.8,
    )
    texture_fit = _semantic_texture_fit(semantic_texture)
    base_fit = (
        within_frequency
        and texture_fit
        and not copy_risk["exact"]
        and brightness_resolution["passes"]
    )
    register_enforcement = _run_register_enforcement(run_dir)
    promotion = apply_register_promotion_policy(
        base_fit,
        register_enforcement,
    )
    diagnostic_fit = bool(promotion["passes"])
    fit = diagnostic_fit and promotion_eligible
    status = promotion["status"] if fit else "completed_unfit"
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "passes": fit,
        "candidate_use": candidate_use,
        "diagnostic_fit": diagnostic_fit,
        "base_quality_passes": promotion["base_quality_passes"],
        "register_enforcement": register_enforcement,
        "descriptor_count": len(descriptors),
        "descriptor_neighborhood_pass_count": sum(
            bool(item["within_neighborhood"]) for item in descriptors
        ),
        "descriptors": descriptors,
        "semantic_texture": semantic_texture,
        "texture_fit": texture_fit,
        "texture_alignment": texture_alignment,
        "controls": measured_controls,
        "brightness_resolution": brightness_resolution,
        "attack_frequency": {
            "source": frequency_contract["source"],
            "requested_normalized": float(
                prompt_target["controls"]["attack_frequency"]
            ),
            "raw_groups_per_second": attack_frequency,
            "target_groups_per_second": frequency_contract["target"],
            "candidate_minimum": frequency_contract["candidate_minimum"],
            "candidate_maximum": frequency_contract["candidate_maximum"],
            "within_candidate_range": within_frequency,
            "group_tolerance_ms": frequency_contract["group_tolerance_ms"],
            "target_score_attack_group_count": target_attack_group_count,
            "score_attack_group_count": score_attack_group_count,
            "rendered_attack_group_count": rendered_attack_group_count,
            "score_group_residual": (
                score_attack_group_count - target_attack_group_count
                if score_attack_group_count is not None
                else None
            ),
            "rendered_group_residual": rendered_attack_group_count
            - target_attack_group_count,
            "attack_group_count": rendered_attack_group_count,
        },
        "copy_risk": copy_risk,
        "promoted": fit,
    }
    atomic_write_json(run_dir / "outputs/candidate-evaluation.json", result)
    store = _run_store(run_dir)
    outputs: dict[str, Any] = {
        "status": status,
        "passes": fit,
        "candidate_use": candidate_use,
        "diagnostic_fit": diagnostic_fit,
        "base_quality_passes": promotion["base_quality_passes"],
        "evaluation_sha256": sha256_file(run_dir / "outputs/candidate-evaluation.json"),
        "promoted": fit,
        "staged_smf_sha256": sha256_file(smf_path),
    }
    if fit:
        final_midi = store.promote_file(
            smf_path,
            run_dir / "outputs/final.mid",
            sha256_file(smf_path),
        )
        final_xml = store.promote_file(
            musicxml_path,
            run_dir / "outputs/final.musicxml",
            sha256_file(musicxml_path),
        )
        outputs.update(
            {
                "smf_path": str(final_midi.relative_to(run_dir)),
                "smf_sha256": sha256_file(final_midi),
                "musicxml_path": str(final_xml.relative_to(run_dir)),
                "musicxml_sha256": sha256_file(final_xml),
            }
        )
    store.record_step(
        "publish-final",
        "completed",
        {
            "staged_smf": sha256_file(smf_path),
            "rendered_output_validation": sha256_json(rendered_output_validation),
            "prompt_target": sha256_json(prompt_target),
            "evaluation_inputs": sha256_json(evaluation_records),
        },
        outputs,
    )
    manifest = _read_json(run_dir / "manifest.json")
    manifest.update(
        {
            "status": status,
            "passes": fit,
            "base_quality_passes": promotion["base_quality_passes"],
            "promoted": fit,
            "attack_frequency": result["attack_frequency"],
            "copy_risk": copy_risk,
            "descriptor_neighborhood_pass_count": result[
                "descriptor_neighborhood_pass_count"
            ],
        }
    )
    atomic_write_json(run_dir / "manifest.json", manifest)
    return result


def _verify_prepared_drift(project_root: Path, run_dir: Path) -> None:
    spec = _read_json(run_dir / "run-spec.json")
    input_hashes = spec.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise WholeScoreLiveRunError("prepared input hashes are missing")
    _verify_file_records(project_root, input_hashes["generation_inputs"])
    implementations = spec.get("implementation_hashes")
    if not isinstance(implementations, dict):
        raise WholeScoreLiveRunError("prepared implementation hashes are missing")
    for relative, expected in implementations.items():
        path = project_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise WholeScoreLiveRunError(f"prepared implementation drifted: {relative}")


def execute_prepared_whole_score_live_run(
    project_root: Path,
    run_dir: Path,
    runner: WholeScoreStageRunner,
) -> dict[str, Any]:
    """準備済みrunを最大5呼び出しで縦断し、自動再試行せず結果を確定する。"""

    project_root = Path(project_root).resolve()
    run_dir = Path(run_dir).resolve()
    manifest = _read_json(run_dir / "manifest.json")
    if manifest.get("status") != "prepared":
        return manifest
    try:
        _verify_prepared_drift(project_root, run_dir)
        skeleton = execute_harmony_stage(project_root, run_dir, runner)
        melodies = execute_melody_stages(project_root, run_dir, runner, skeleton)
        texture = execute_texture_stage(
            project_root,
            run_dir,
            runner,
            skeleton,
            melodies,
        )
        plan = parse_piece_plan(
            (run_dir / "inputs/piece-plan.dsl").read_text(encoding="utf-8")
        )
        score_quality = evaluate_generic_score_quality(plan, texture.score)
        atomic_write_json(run_dir / "outputs/generic-score-quality.json", score_quality)
        if not score_quality.get("passes"):
            raise WholeScoreLiveRunError(
                "generated whole score failed generic score quality before "
                f"performance: {score_quality.get('failures')}"
            )
        execute_performance_stage(
            project_root,
            run_dir,
            runner,
            texture.score,
        )
        result = evaluate_staged_candidate(project_root, run_dir)
        manifest = _read_json(run_dir / "manifest.json")
        manifest["confirmed_external_call_count"] = runner.call_number
        manifest["result"] = {
            "status": result["status"],
            "promoted": result["promoted"],
        }
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "passes": False,
                "promoted": False,
                "confirmed_external_call_count": runner.call_number,
                "error": {"type": type(error).__name__, "detail": str(error)},
            }
        )
        try:
            _run_store(run_dir).record_step(
                "run-failure",
                "failed",
                {"run_spec": sha256_file(run_dir / "run-spec.json")},
                manifest["error"],
            )
        except Exception as state_error:
            manifest["state_record_error"] = {
                "type": type(state_error).__name__,
                "detail": str(state_error),
            }
    atomic_write_json(run_dir / "manifest.json", manifest)
    return manifest


def create_default_whole_score_runner(
    project_root: Path,
    run_dir: Path,
    *,
    maximum_external_calls: int | None = None,
) -> WholeScoreStageRunner:
    """分離した作業領域で上限5回のCodex実行器を作る。"""

    project_root = Path(project_root).resolve()
    run_dir = Path(run_dir).resolve()
    prepared_ceiling = _run_maximum_external_calls(run_dir)
    active_ceiling = (
        prepared_ceiling
        if maximum_external_calls is None
        else maximum_external_calls
    )
    if not 1 <= active_ceiling <= prepared_ceiling:
        raise WholeScoreLiveRunError("external call ceiling is invalid")
    return CodexExecRunner(
        run_store=RunStore(run_dir, max_calls=active_ceiling),
        schema_path=project_root / "schemas/codex-composition-response.schema.json",
        model=MODEL_CONFIG["model"],
        reasoning_effort=MODEL_CONFIG["reasoning_effort"],
        working_directory=isolated_codex_working_directory(),
        timeout_seconds=MODEL_CONFIG["timeout_seconds"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        result = prepare_whole_score_live_run(arguments.project_root, arguments.run_dir)
        success = result["status"] == "prepared"
    else:
        runner = create_default_whole_score_runner(
            arguments.project_root,
            arguments.run_dir,
        )
        result = execute_prepared_whole_score_live_run(
            arguments.project_root,
            arguments.run_dir,
            runner,
        )
        success = result.get("status") == "completed_fit"
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if success else 1


def _ordered_leaves(plan: PiecePlan) -> tuple[PlanNode, ...]:
    nodes = {item.node_id: item for item in plan.nodes}
    children: dict[str, list[PlanNode]] = {item.node_id: [] for item in plan.nodes}
    for node in plan.nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node)

    def visit(node: PlanNode) -> tuple[PlanNode, ...]:
        descendants = sorted(children[node.node_id], key=lambda item: item.order)
        if not descendants:
            return (node,)
        return tuple(leaf for child in descendants for leaf in visit(child))

    return visit(nodes[plan.root_node_id])


def _maximum_material_note_events(length_units: int, divisions: int) -> int:
    per_stage = min(64, 6 * math.ceil(length_units / divisions))
    return per_stage * 2


def rescale_harmonic_material(
    item: WholeHarmonicMaterialDraftV0,
    new_length: int,
) -> WholeHarmonicMaterialDraftV0:
    """和声内容と時間比を保ち、素材の整数時間格子だけを拡張する。"""

    if new_length < item.length_units:
        raise WholeScoreLiveRunError("harmonic material length cannot shrink")
    ideals = [
        Fraction(event.duration_units * new_length, item.length_units)
        for event in item.draft.events
    ]
    durations = [value.numerator // value.denominator for value in ideals]
    missing = new_length - sum(durations)
    order = sorted(
        range(len(ideals)),
        key=lambda index: (-(ideals[index] - durations[index]), index),
    )
    for index in order[:missing]:
        durations[index] += 1
    cursor = 0
    events = []
    for source, duration in zip(item.draft.events, durations, strict=True):
        if duration <= 0:
            raise WholeScoreLiveRunError(
                "normalized harmony duration must be positive"
            )
        events.append(
            HarmonicEventDraft(
                cursor,
                duration,
                source.root_pitch_class,
                source.quality,
            )
        )
        cursor += duration
    return WholeHarmonicMaterialDraftV0(
        new_length,
        type(item.draft)(tuple(events)),
    )


def _harmonic_content_invariants(
    source: tuple[WholeHarmonicMaterialDraftV0, ...],
    effective: tuple[WholeHarmonicMaterialDraftV0, ...],
) -> dict[str, bool]:
    return {
        "event_count_unchanged": all(
            len(before.draft.events) == len(after.draft.events)
            for before, after in zip(source, effective, strict=True)
        ),
        "event_order_unchanged": all(
            [
                (event.root_pitch_class, event.quality)
                for event in before.draft.events
            ]
            == [
                (event.root_pitch_class, event.quality)
                for event in after.draft.events
            ]
            for before, after in zip(source, effective, strict=True)
        ),
        "root_pitch_class_unchanged": all(
            [event.root_pitch_class for event in before.draft.events]
            == [event.root_pitch_class for event in after.draft.events]
            for before, after in zip(source, effective, strict=True)
        ),
        "quality_unchanged": all(
            [event.quality for event in before.draft.events]
            == [event.quality for event in after.draft.events]
            for before, after in zip(source, effective, strict=True)
        ),
    }


def _harmonic_capacity_skeleton(
    plan: PiecePlan,
    drafts: tuple[WholeHarmonicMaterialDraftV0, ...],
) -> HarmonicSkeletonV0:
    return assemble_whole_score_skeleton(
        "harmonic-capacity-normalization-v1",
        plan,
        tuple((item.length_units, item.draft) for item in drafts),
    )


def _harmonic_capacity_diagnostic(
    plan: PiecePlan,
    source: tuple[WholeHarmonicMaterialDraftV0, ...],
    effective: tuple[WholeHarmonicMaterialDraftV0, ...],
    *,
    status: str,
    minimum_attack_group_count: int,
    reserve_ending_hold: bool,
) -> dict[str, Any]:
    context = build_whole_score_context(plan)
    source_skeleton = _harmonic_capacity_skeleton(plan, source)
    effective_skeleton = _harmonic_capacity_skeleton(plan, effective)
    source_capacity = maximum_attack_group_capacity(
        plan,
        source_skeleton,
        reserve_ending_hold=reserve_ending_hold,
    )
    effective_capacity = maximum_attack_group_capacity(
        plan,
        effective_skeleton,
        reserve_ending_hold=reserve_ending_hold,
    )
    source_ending = validate_early_ending(plan, source_skeleton)
    effective_ending = validate_early_ending(plan, effective_skeleton)
    invariants = _harmonic_content_invariants(source, effective)
    if not all(invariants.values()):
        raise WholeScoreLiveRunError("harmonic content changed during normalization")
    factors = [
        after.length_units / before.length_units
        for before, after in zip(source, effective, strict=True)
    ]
    return {
        "schema_version": 1,
        "status": status,
        "minimum_attack_group_count": minimum_attack_group_count,
        "reserve_ending_hold": reserve_ending_hold,
        "allocation_policy": "minimum-next-unit-ratio-then-material-order-v1",
        "source_capacity": source_capacity,
        "source_early_ending": source_ending,
        "effective_capacity": effective_capacity,
        "effective_early_ending": effective_ending,
        "minimum_expansion_factor": min(factors),
        "maximum_expansion_factor": max(factors),
        "materials": [
            {
                "material_id": material_id,
                "source_length_units": before.length_units,
                "effective_length_units": after.length_units,
                "expansion_factor": factor,
            }
            for material_id, before, after, factor in zip(
                context.material_ids,
                source,
                effective,
                factors,
                strict=True,
            )
        ],
        "content_invariants": invariants,
    }


def normalize_harmonic_capacity(
    plan: PiecePlan,
    source: tuple[WholeHarmonicMaterialDraftV0, ...],
    *,
    minimum_attack_group_count: int,
    reserve_ending_hold: bool = True,
) -> tuple[tuple[WholeHarmonicMaterialDraftV0, ...], dict[str, Any]]:
    """音楽内容を保ち、後段に必要な整数発音位置だけを確保する。"""

    if minimum_attack_group_count <= 0:
        raise WholeScoreLiveRunError("minimum capacity must be positive")
    context = build_whole_score_context(plan)
    if len(source) != len(context.material_ids):
        raise WholeScoreLiveRunError(
            "harmonic material count does not match PiecePlan"
        )
    source_skeleton = _harmonic_capacity_skeleton(plan, source)
    source_capacity = maximum_attack_group_capacity(
        plan,
        source_skeleton,
        reserve_ending_hold=reserve_ending_hold,
    )
    validate_early_ending(plan, source_skeleton)
    if source_capacity["maximum_attack_group_count"] >= minimum_attack_group_count:
        return source, _harmonic_capacity_diagnostic(
            plan,
            source,
            source,
            status="unchanged",
            minimum_attack_group_count=minimum_attack_group_count,
            reserve_ending_hold=reserve_ending_hold,
        )

    original_lengths = [item.length_units for item in source]
    lengths = list(original_lengths)
    current_count = source_capacity["maximum_attack_group_count"]
    while current_count < minimum_attack_group_count:
        candidates = [
            (Fraction(current + 1, original), index)
            for index, (original, current) in enumerate(
                zip(original_lengths, lengths, strict=True)
            )
        ]
        _, selected = min(candidates)
        lengths[selected] += 1
        provisional = tuple(
            rescale_harmonic_material(item, length)
            for item, length in zip(source, lengths, strict=True)
        )
        current_count = maximum_attack_group_capacity(
            plan,
            _harmonic_capacity_skeleton(plan, provisional),
            reserve_ending_hold=reserve_ending_hold,
        )["maximum_attack_group_count"]

    effective = tuple(
        rescale_harmonic_material(item, length)
        for item, length in zip(source, lengths, strict=True)
    )
    diagnostic = _harmonic_capacity_diagnostic(
        plan,
        source,
        effective,
        status="normalized",
        minimum_attack_group_count=minimum_attack_group_count,
        reserve_ending_hold=reserve_ending_hold,
    )
    if (
        diagnostic["effective_capacity"]["maximum_attack_group_count"]
        < minimum_attack_group_count
    ):
        raise WholeScoreLiveRunError("harmonic capacity normalization failed")
    return effective, diagnostic


def maximum_attack_group_capacity(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    *,
    duration_ms: int = WHOLE_SCORE_TARGET_DURATION_MS,
    reserve_ending_hold: bool = True,
) -> dict[str, Any]:
    """全出現の旋律・伴奏note event上限から30 ms群の絶対上限を返す。"""

    if duration_ms <= 0:
        raise WholeScoreLiveRunError("duration must be positive")
    validate_harmonic_skeleton(plan, skeleton)
    materials = {item.material_id: item for item in skeleton.materials}
    leaves = _ordered_leaves(plan)
    per_material = {
        material_id: _maximum_material_note_events(
            material.length_units,
            skeleton.divisions,
        )
        for material_id, material in materials.items()
    }
    capacity_records = (
        material_attack_group_capacities(
            plan,
            skeleton,
            duration_ms=duration_ms,
        )
        if reserve_ending_hold
        else {
            material_id: {
                "attack_group_capacity": material.length_units,
                "reserved_ending_units": 0,
            }
            for material_id, material in materials.items()
        }
    )
    per_material_groups = {
        material_id: record["attack_group_capacity"]
        for material_id, record in capacity_records.items()
    }
    maximum_groups = sum(
        per_material_groups[leaf.score_material_id] for leaf in leaves
    )
    maximum_note_events = sum(
        per_material[leaf.score_material_id] for leaf in leaves
    )
    duration_seconds = duration_ms / 1_000
    return {
        "basis": (
            "ending_reserved_integer_score_onset_count"
            if reserve_ending_hold
            else "expanded_integer_score_onset_count"
        ),
        "duration_ms": duration_ms,
        "occurrence_count": len(leaves),
        "maximum_attack_group_count": maximum_groups,
        "maximum_attack_groups_per_second": maximum_groups / duration_seconds,
        "maximum_note_event_count": maximum_note_events,
        "per_material_maximum_attack_group_count": per_material_groups,
        "per_material_attack_group_capacity": per_material_groups,
        "per_material_reserved_ending_units": {
            material_id: record["reserved_ending_units"]
            for material_id, record in capacity_records.items()
        },
        "per_material_maximum_note_event_count": per_material,
    }


def validate_early_ending(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    *,
    minimum_hold_ms: int = 2_000,
    duration_ms: int = WHOLE_SCORE_TARGET_DURATION_MS,
) -> dict[str, Any]:
    """和声段階で最終素材の主和音と名目保持容量を検査する。"""

    if minimum_hold_ms <= 0 or duration_ms <= 0:
        raise WholeScoreLiveRunError("ending durations must be positive")
    validate_harmonic_skeleton(plan, skeleton)
    leaves = _ordered_leaves(plan)
    if not leaves:
        raise WholeScoreLiveRunError("PiecePlan has no ending occurrence")
    final_leaf = leaves[-1]
    if final_leaf.role != "release" or final_leaf.score_material_id is None:
        raise WholeScoreLiveRunError("final occurrence must be a dedicated release")
    materials = {item.material_id: item for item in skeleton.materials}
    final_material = materials[final_leaf.score_material_id]
    if len(final_material.harmonies) != 1:
        raise WholeScoreLiveRunError("ending must contain one tonic harmony")
    harmony = final_material.harmonies[0]
    if harmony.root_pitch_class != plan.tonal_center or harmony.quality != plan.mode:
        raise WholeScoreLiveRunError("ending tonic root or mode is invalid")
    total_units = sum(materials[leaf.score_material_id].length_units for leaf in leaves)
    nominal_capacity_ms = duration_ms * final_material.length_units / total_units
    if nominal_capacity_ms < minimum_hold_ms:
        raise WholeScoreLiveRunError("ending hold capacity is insufficient")
    return {
        "status": "pass",
        "final_node_id": final_leaf.node_id,
        "final_material_id": final_material.material_id,
        "root_pitch_class": harmony.root_pitch_class,
        "quality": harmony.quality,
        "nominal_capacity_ms": nominal_capacity_ms,
        "minimum_hold_ms": minimum_hold_ms,
    }


def _ending_note_requirements(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    *,
    minimum_hold_ms: int = 2_000,
    duration_ms: int = WHOLE_SCORE_TARGET_DURATION_MS,
) -> dict[str, Any]:
    materials = {item.material_id: item for item in skeleton.materials}
    total_units = sum(
        materials[leaf.score_material_id].length_units for leaf in _ordered_leaves(plan)
    )
    minimum_duration = math.ceil(total_units * minimum_hold_ms / duration_ms)
    leaves = _ordered_leaves(plan)
    if not leaves or leaves[-1].score_material_id is None:
        raise WholeScoreLiveRunError("ending material is missing")
    final_material = materials[leaves[-1].score_material_id]
    return {
        "minimum_final_duration_units": minimum_duration,
        "required_final_duration_units": minimum_duration,
        "required_final_attack_units": final_material.length_units - minimum_duration,
        "required_final_pitch_classes": [
            plan.tonal_center,
            (plan.tonal_center + 7) % 12,
        ],
    }


def normalize_melody_ending(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    drafts: tuple[Any, ...],
    *,
    allowed_pitch_range: tuple[int, int] | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """最終旋律和音を伴奏と共有する終止打鍵へそろえる。"""

    context = build_whole_score_context(plan)
    if len(drafts) != len(context.material_ids) or len(context.release_material_ids) != 1:
        raise WholeScoreLiveRunError("shared ending requires one release melody")
    final_material_id = context.release_material_ids[0]
    final_index = context.material_ids.index(final_material_id)
    final_draft = drafts[final_index]
    if not final_draft.events:
        raise WholeScoreLiveRunError("shared ending melody is empty")
    requirements = _ending_note_requirements(plan, skeleton)
    required_attack = int(requirements["required_final_attack_units"])
    required_duration = int(requirements["required_final_duration_units"])
    source_attack = max(event.at_units for event in final_draft.events)
    final_indices = tuple(
        index
        for index, event in enumerate(final_draft.events)
        if event.at_units == source_attack
    )
    if any(
        event.at_units < required_attack < event.at_units + event.duration_units
        for index, event in enumerate(final_draft.events)
        if index not in final_indices
    ):
        raise WholeScoreLiveRunError("melody crosses the shared ending attack")
    normalized_events = tuple(
        replace(event, at_units=required_attack, duration_units=required_duration)
        if index in final_indices
        else event
        for index, event in enumerate(final_draft.events)
    )
    normalized = list(drafts)
    normalized[final_index] = replace(final_draft, events=normalized_events)
    timing_changed_indices = tuple(
        index
        for index in final_indices
        if final_draft.events[index] != normalized_events[index]
    )
    pitch_transposition = None
    assignment = None
    maximum_leap_before = _maximum_representative_melody_leap(final_draft)
    maximum_leap_after = _maximum_representative_melody_leap(normalized[final_index])
    if allowed_pitch_range is not None:
        lower, upper = allowed_pitch_range
        if not 21 <= lower <= upper <= 108:
            raise WholeScoreLiveRunError("shared ending pitch range is invalid")
        base_payload = _assemble_ending_candidate(plan, skeleton, tuple(normalized))
        base_ids = _payload_event_ids(base_payload)
        assignment_candidates = shared_ending_pitch_assignments(
            foreground_pitches=tuple(
                event.pitch
                for event in normalized[final_index].events
                if event.at_units == required_attack
            ),
            foreground_voice=final_draft.foreground_voice,
            root_pitch_class=plan.tonal_center,
            allowed_pitch_range=allowed_pitch_range,
        )
        if assignment_candidates:
            assignment = assignment_candidates[0]
        else:
            choices: list[tuple[tuple[Any, ...], tuple[Any, ...], dict[str, Any]]] = []
            source_release = normalized[final_index]
            source_maximum = _maximum_representative_melody_leap(source_release)
            boundaries = sorted({event.at_units for event in source_release.events})
            for suffix_start in boundaries:
                for semitones in range(-84, 85, 12):
                    if semitones == 0:
                        continue
                    candidate_events = tuple(
                        replace(event, pitch=event.pitch + semitones)
                        if event.at_units >= suffix_start
                        else event
                        for event in source_release.events
                    )
                    if any(
                        not lower <= event.pitch <= upper
                        for event in candidate_events
                    ):
                        continue
                    candidate_release = replace(source_release, events=candidate_events)
                    boundary_leap = _suffix_boundary_leap(
                        candidate_release, suffix_start
                    )
                    if suffix_start != boundaries[0] and (
                        boundary_leap is None or boundary_leap > 7
                    ):
                        continue
                    candidate_maximum = _maximum_representative_melody_leap(
                        candidate_release
                    )
                    if candidate_maximum > max(source_maximum, 7):
                        continue
                    candidate_assignments = shared_ending_pitch_assignments(
                        foreground_pitches=tuple(
                            event.pitch
                            for event in candidate_events
                            if event.at_units == required_attack
                        ),
                        foreground_voice=final_draft.foreground_voice,
                        root_pitch_class=plan.tonal_center,
                        allowed_pitch_range=allowed_pitch_range,
                    )
                    if not candidate_assignments:
                        continue
                    candidate_drafts = tuple(
                        candidate_release if index == final_index else draft
                        for index, draft in enumerate(normalized)
                    )
                    try:
                        payload = _assemble_ending_candidate(
                            plan, skeleton, candidate_drafts
                        )
                    except WholeScoreLiveRunError:
                        continue
                    if _payload_event_ids(payload) != base_ids:
                        continue
                    changed_count = sum(
                        event.at_units >= suffix_start
                        for event in source_release.events
                    )
                    rank = (
                        changed_count,
                        changed_count * abs(semitones),
                        abs(semitones) // 12,
                        suffix_start,
                    )
                    choices.append(
                        (
                            rank,
                            candidate_drafts,
                            {
                                "suffix_start_units": suffix_start,
                                "semitones": semitones,
                                "changed_event_count": changed_count,
                                "boundary_leap": boundary_leap,
                                "maximum_representative_leap_before": source_maximum,
                                "maximum_representative_leap_after": candidate_maximum,
                                "required_final_pitch_assignment": candidate_assignments[0],
                            },
                        )
                    )
            if not choices:
                raise WholeScoreLiveRunError(
                    "shared ending accompaniment is not jointly placeable"
                )
            _, selected_drafts, pitch_transposition = min(
                choices, key=lambda item: item[0]
            )
            normalized = list(selected_drafts)
            assignment = pitch_transposition["required_final_pitch_assignment"]
            maximum_leap_after = int(
                pitch_transposition["maximum_representative_leap_after"]
            )
    pitch_changed_indices = tuple(
        index
        for index, (before, after) in enumerate(
            zip(final_draft.events, normalized[final_index].events, strict=True)
        )
        if before.pitch != after.pitch
    )
    changed_indices = sorted(set(timing_changed_indices) | set(pitch_changed_indices))
    changed = bool(changed_indices)
    source_durations = sorted(
        {final_draft.events[index].duration_units for index in final_indices}
    )
    return tuple(normalized), {
        "schema_version": 2,
        "status": "normalized" if changed else "unchanged",
        "material_id": final_material_id,
        "source_attack_units": source_attack,
        "required_final_attack_units": required_attack,
        "source_duration_units": (
            source_durations[0] if len(source_durations) == 1 else source_durations
        ),
        "required_final_duration_units": required_duration,
        "changed_event_indices": changed_indices,
        "changed_event_count": len(changed_indices),
        "pitch_changed_event_indices": list(pitch_changed_indices),
        "pitch_transposition": pitch_transposition,
        "maximum_representative_leap_before": maximum_leap_before,
        "maximum_representative_leap_after": maximum_leap_after,
        "required_final_pitch_assignment": assignment,
        "allowed_pitch_range": (
            list(allowed_pitch_range) if allowed_pitch_range is not None else None
        ),
        "minimum_hold_ms": minimum_hold_ms_from_units(
            plan, skeleton, required_duration
        ),
    }


def shared_ending_pitch_assignments(
    *,
    foreground_pitches: tuple[int, ...],
    foreground_voice: str,
    root_pitch_class: int,
    allowed_pitch_range: tuple[int, int],
) -> tuple[dict[str, Any], ...]:
    """共同終止の根音・完全五度を安全に置ける実音高組を返す。"""

    if not foreground_pitches or foreground_voice not in {"upper", "lower"}:
        raise WholeScoreLiveRunError("shared ending foreground is invalid")
    lower, upper = allowed_pitch_range
    if not 21 <= lower <= upper <= 108 or not 0 <= root_pitch_class <= 11:
        raise WholeScoreLiveRunError("shared ending pitch assignment is invalid")
    degree_classes = {
        "root": root_pitch_class,
        "fifth": (root_pitch_class + 7) % 12,
    }
    candidates = {
        degree: tuple(
            pitch
            for pitch in range(lower, upper + 1)
            if pitch % 12 == pitch_class
        )
        for degree, pitch_class in degree_classes.items()
    }
    boundary = min(foreground_pitches) if foreground_voice == "upper" else max(
        foreground_pitches
    )
    results: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for root_pitch in candidates["root"]:
        for fifth_pitch in candidates["fifth"]:
            accompaniment = (root_pitch, fifth_pitch)
            if root_pitch == fifth_pitch:
                continue
            if foreground_voice == "upper" and any(
                pitch >= boundary for pitch in accompaniment
            ):
                continue
            if foreground_voice == "lower" and any(
                pitch <= boundary for pitch in accompaniment
            ):
                continue
            if not _pitches_pass_shared_ending_low_spacing(
                (*foreground_pitches, *accompaniment)
            ):
                continue
            zones = {
                "root": _suggested_register_zone(root_pitch),
                "fifth": _suggested_register_zone(fifth_pitch),
            }
            distance = sum(abs(boundary - pitch) for pitch in accompaniment)
            zone_distance = sum(
                abs(REGISTER_ZONES[zones[degree]][2] - pitch)
                for degree, pitch in (
                    ("root", root_pitch),
                    ("fifth", fifth_pitch),
                )
            )
            assignment = {
                "root_pitch_class": root_pitch_class,
                "foreground_voice": foreground_voice,
                "foreground_pitches": list(foreground_pitches),
                "pitches": {"root": root_pitch, "fifth": fifth_pitch},
                "suggested_register_zones": zones,
                "minimum_required_new_accompaniment_attack_count": 2,
                "constraints": [
                    "allowed_pitch_range",
                    "foreground_voice_order",
                    "minimum_low_spacing_7_semitones",
                    "distinct_root_and_fifth",
                ],
            }
            results.append(
                ((distance, zone_distance, root_pitch, fifth_pitch), assignment)
            )
    return tuple(item for _, item in sorted(results, key=lambda item: item[0]))


def _suggested_register_zone(pitch: int) -> str:
    candidates = [
        (abs(center - pitch), index, zone)
        for index, (zone, (lower, upper, center)) in enumerate(
            REGISTER_ZONES.items()
        )
        if lower <= pitch <= upper
    ]
    if not candidates:
        raise WholeScoreLiveRunError("shared ending pitch has no register zone")
    return min(candidates)[2]


def _pitches_pass_shared_ending_low_spacing(pitches: Sequence[int]) -> bool:
    ordered = sorted(set(pitches))
    return (
        len(ordered) < 2
        or ordered[0] >= LOW_PITCH_BOUNDARY
        or ordered[1] - ordered[0] >= MINIMUM_LOW_SPACING_SEMITONES
    )


def _representative_melody_pitches(draft: Any) -> tuple[tuple[int, int], ...]:
    grouped: dict[int, list[int]] = {}
    for event in draft.events:
        grouped.setdefault(event.at_units, []).append(event.pitch)
    select = max if draft.foreground_voice == "upper" else min
    return tuple(
        (at_units, select(pitches)) for at_units, pitches in sorted(grouped.items())
    )


def _maximum_representative_melody_leap(draft: Any) -> int:
    pitches = [pitch for _, pitch in _representative_melody_pitches(draft)]
    return max(
        (abs(after - before) for before, after in pairwise(pitches)),
        default=0,
    )


def _suffix_boundary_leap(draft: Any, suffix_start_units: int) -> int | None:
    representatives = _representative_melody_pitches(draft)
    index = next(
        (
            index
            for index, (at_units, _) in enumerate(representatives)
            if at_units == suffix_start_units
        ),
        None,
    )
    if index is None or index == 0:
        return None
    return abs(representatives[index][1] - representatives[index - 1][1])


def _assemble_ending_candidate(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    drafts: tuple[Any, ...],
) -> ScorePayloadV0:
    try:
        return assemble_whole_score_melodies(
            "shared-ending-normalization", plan, skeleton, drafts
        )
    except ValueError as error:
        raise WholeScoreLiveRunError(str(error)) from error


def _payload_event_ids(payload: ScorePayloadV0) -> tuple[str, ...]:
    return tuple(
        note.event_id for material in payload.materials for note in material.notes
    )


def minimum_hold_ms_from_units(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    duration_units: int,
) -> int:
    materials = {item.material_id: item for item in skeleton.materials}
    total_units = sum(
        materials[leaf.score_material_id].length_units for leaf in _ordered_leaves(plan)
    )
    return duration_units * WHOLE_SCORE_TARGET_DURATION_MS // total_units


def validate_melody_ending(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    melodies: ScorePayloadV0,
) -> dict[str, Any]:
    """最終旋律が終止保持余白の契約を満たすことを検査する。"""

    leaves = _ordered_leaves(plan)
    if not leaves or leaves[-1].role != "release":
        raise WholeScoreLiveRunError("ending melody requires a final release")
    final_material_id = leaves[-1].score_material_id
    if final_material_id is None:
        raise WholeScoreLiveRunError("ending melody material is missing")
    skeleton_materials = {item.material_id: item for item in skeleton.materials}
    melody_materials = {item.material_id: item for item in melodies.materials}
    try:
        material = skeleton_materials[final_material_id]
        melody = melody_materials[final_material_id]
    except KeyError as error:
        raise WholeScoreLiveRunError("ending melody material is missing") from error
    requirements = _ending_note_requirements(plan, skeleton)
    minimum_duration = int(requirements["minimum_final_duration_units"])
    latest_attack = material.length_units - minimum_duration
    final_attack = max((note.at_units for note in melody.notes), default=None)
    chord = tuple(
        note for note in melody.notes if final_attack is not None and note.at_units == final_attack
    )
    tonic = plan.tonal_center
    valid = (
        final_attack is not None
        and final_attack == latest_attack
        and bool(chord)
        and all(note.pitch % 12 == tonic for note in chord)
        and all(note.duration_units == minimum_duration for note in chord)
        and all(note.at_units + note.duration_units == material.length_units for note in chord)
        and not any(
            note.at_units < final_attack < note.at_units + note.duration_units
            for note in melody.notes
        )
    )
    if not valid:
        raise WholeScoreLiveRunError("ending melody does not satisfy final hold contract")
    return {
        "status": "pass",
        "material_id": final_material_id,
        "final_attack_units": final_attack,
        "latest_final_attack_units": latest_attack,
        "minimum_final_duration_units": minimum_duration,
        "reserved_ending_units": minimum_duration - 1,
        "tonal_center": tonic,
    }


if __name__ == "__main__":  # pragma: no cover - CLI入口
    raise SystemExit(main())
