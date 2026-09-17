"""匿名参照目標から3段IRを生成し、検証済みMusicXMLとSMFへ確定する。"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from llm_musical_composer.generation_intent import creative_targets_for_stage
from llm_musical_composer.generic_pipeline_quality import (
    evaluate_generic_piece_plan_quality,
    evaluate_generic_pipeline_quality,
    evaluate_generic_score_quality,
)
from llm_musical_composer.performance_pipeline import (
    PerformanceSpec,
    PiecePlan,
    PipelineValidationError,
    RenderedPerformance,
    ScoreSpec,
    ordered_leaf_schedule,
    render_musicxml,
    render_performance,
    render_performance_smf,
    validate_piece_plan,
    validate_pipeline,
    validate_score_materials,
    validate_score_spec,
)
from llm_musical_composer.pipeline_dsl import (
    PipelineDslError,
    dump_performance_spec,
    dump_piece_plan,
    dump_score_spec,
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.pipeline_prompts import build_stage_prompt
from llm_musical_composer.run_state import (
    RunLock,
    RunStore,
    StateConflictError,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
)
from llm_musical_composer.smf_notes import load_smf_notes
from llm_musical_composer.tonal_hierarchy import evaluate_piece_plan_tonal_hierarchy

StopAfter = Literal["piece_plan", "score_spec", "performance_spec"]
MAX_MATERIALS_PER_BATCH = 8


class StageResponseValidationError(ValueError):
    """モデルが返した段階応答と、要求した段階契約が一致しない。"""


class StageRunner(Protocol):
    """段階別プロンプトを実行する最小インターフェース。"""

    def run(
        self,
        step_id: str,
        prompt: str,
        input_hashes: Mapping[str, str] | None = None,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class StagedPipelineResult:
    """途中停止または最終確定した生成結果。"""

    status: str
    stopped_after: str | None
    piece_plan_path: Path
    score_spec_path: Path | None
    performance_spec_path: Path | None
    musicxml_path: Path | None
    smf_path: Path | None
    quality: dict[str, Any] | None


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def staged_pipeline_fingerprints() -> dict[str, str]:
    """段階生成の判断と状態確定に関与するローカル実装を固定する。"""

    root = Path(__file__).resolve().parent
    project_root = root.parents[1]
    paths = {
        "orchestrator": Path(__file__).resolve(),
        "generation_intent": root / "generation_intent.py",
        "pipeline": root / "performance_pipeline.py",
        "dsl": root / "pipeline_dsl.py",
        "prompts": root / "pipeline_prompts.py",
        "quality": root / "generic_pipeline_quality.py",
        "recurrence_analysis": root / "recurrence_analysis.py",
        "recurrence_quality": root / "recurrence_quality.py",
        "run_state": root / "run_state.py",
        "smf_notes": root / "smf_notes.py",
        "tonal_hierarchy": root / "tonal_hierarchy.py",
        "reference_variance_smoke": root / "reference_variance_smoke.py",
        "prompt_piece_plan": project_root / "prompts" / "pipeline-piece-plan.md",
        "prompt_score_spec": project_root / "prompts" / "pipeline-score-spec.md",
        "prompt_performance_spec": project_root / "prompts" / "pipeline-performance-spec.md",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _anonymous_request(normalized_request: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in normalized_request.items() if key != "reference"}


def _stage_target(prompt_target: Mapping[str, object], stage: str) -> dict[str, object]:
    controls = prompt_target.get("controls", {})
    if stage == "piece_plan":
        semantic_targets = prompt_target.get("semantic_targets", {})
        piece_targets = (
            semantic_targets.get("piece_plan", [])
            if isinstance(semantic_targets, Mapping)
            else []
        )
        specified = [
            item
            for item in piece_targets
            if isinstance(item, Mapping)
            and item.get("id") == "tonal_hierarchy"
            and item.get("status") == "specified"
        ]
        result: dict[str, object] = {
            "controls": controls,
            "piece_plan": prompt_target.get("piece_plan", {}),
            "semantic_targets": specified,
        }
        creative = creative_targets_for_stage(prompt_target, "piece_plan")
        if creative:
            result["creative_targets"] = creative
        return result
    stage_targets = prompt_target.get("stage_targets", {})
    if not isinstance(stage_targets, Mapping):
        stage_targets = {}
    return {
        "controls": controls,
        "stage_targets": {
            stage: stage_targets.get(stage, []),
            "rendered_surface": stage_targets.get("rendered_surface", []),
        },
    }


def _input_hashes(
    *,
    prompt: str,
    target: Mapping[str, object],
    model_config: Mapping[str, object],
    previous_source: str = "",
) -> dict[str, str]:
    result = {
        "prompt": sha256_text(prompt),
        "target": sha256_json(target),
        "model_config": sha256_json(model_config),
        "previous_ir": sha256_text(previous_source),
        **staged_pipeline_fingerprints(),
    }
    creative_targets = target.get("creative_targets")
    if isinstance(creative_targets, list) and creative_targets:
        result["creative_targets"] = sha256_json(creative_targets)
    return result


def _read_step(store: RunStore, step_id: str) -> dict[str, Any] | None:
    path = store.run_dir / "steps" / f"{step_id}.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StateConflictError(f"saved step is not an object: {path}")
    return value


def _quality_input_hashes(
    *, plan_source: str, score_source: str, performance_source: str = ""
) -> dict[str, str]:
    return {
        "piece_plan": sha256_text(plan_source),
        "score_spec": sha256_text(score_source),
        "performance_spec": sha256_text(performance_source),
        **staged_pipeline_fingerprints(),
    }


def _save_quality(
    store: RunStore,
    step_id: str,
    quality_path: Path,
    quality: Mapping[str, object],
    input_hashes: Mapping[str, str],
) -> None:
    store.snapshot_json(quality_path.relative_to(store.run_dir), quality)
    store.record_step(
        step_id,
        "completed" if quality.get("passes") is True else "failed",
        input_hashes,
        {
            "path": str(quality_path.relative_to(store.run_dir)),
            "sha256": sha256_file(quality_path),
        },
    )


def _save_stage_failure(
    store: RunStore,
    step_id: str,
    source: str,
    error: Exception,
    input_hashes: Mapping[str, str],
) -> None:
    relative_path = Path("failures") / f"{step_id}.json"
    path = store.run_dir / relative_path
    failure = {
        "schema_version": 1,
        "step_id": step_id,
        "error_type": type(error).__name__,
        "detail": str(error),
        "composition_source": source,
        "source_sha256": sha256_text(source),
    }
    store.snapshot_json(relative_path, failure)
    store.record_step(
        step_id,
        "failed",
        input_hashes,
        {
            "failure_path": str(relative_path),
            "failure_sha256": sha256_file(path),
        },
    )


def _validate_failure_state(store: RunStore) -> None:
    failure_root = store.run_dir / "failures"
    artifact_ids: set[str] = set()
    for path in sorted(failure_root.glob("*.json")):
        step_id = path.stem
        artifact_ids.add(step_id)
        try:
            failure = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StateConflictError(f"cannot read stage failure: {step_id}") from error
        if not isinstance(failure, dict):
            raise StateConflictError(f"stage failure is not an object: {step_id}")
        source = failure.get("composition_source")
        if (
            failure.get("schema_version") != 1
            or failure.get("step_id") != step_id
            or not isinstance(failure.get("error_type"), str)
            or not isinstance(failure.get("detail"), str)
            or not isinstance(source, str)
            or failure.get("source_sha256") != sha256_text(source)
        ):
            raise StateConflictError(f"stage failure content conflicts: {step_id}")
        step = _read_step(store, step_id)
        outputs = step.get("outputs") if step is not None else None
        relative_path = Path("failures") / path.name
        if (
            step is None
            or step.get("status") != "failed"
            or not isinstance(outputs, Mapping)
            or outputs.get("failure_path") != str(relative_path)
            or outputs.get("failure_sha256") != sha256_file(path)
        ):
            raise StateConflictError(f"stage failure and step must agree: {step_id}")
    for step_path in sorted((store.run_dir / "steps").glob("*.json")):
        step = _read_step(store, step_path.stem)
        assert step is not None
        outputs = step.get("outputs")
        if (
            isinstance(outputs, Mapping)
            and "failure_path" in outputs
            and step_path.stem not in artifact_ids
        ):
            raise StateConflictError(
                f"stage failure and step must coexist: {step_path.stem}"
            )


def _validate_quality_state(store: RunStore) -> None:
    for step_id in ("piece-plan-quality", "score-quality", "performance-quality"):
        path = store.run_dir / "outputs" / f"{step_id}.json"
        step = _read_step(store, step_id)
        if path.is_file() != (step is not None):
            raise StateConflictError(f"quality output and step must coexist: {step_id}")
        if step is None:
            continue
        outputs = step.get("outputs")
        expected = outputs.get("sha256") if isinstance(outputs, Mapping) else None
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise StateConflictError(f"saved quality hash conflicts: {step_id}")
    state = store.rebuild_state()
    if state.get("status") == "failed":
        raise StateConflictError("saved run is failed and cannot be resumed")


def _reuse_source(
    store: RunStore,
    step_id: str,
    output_path: Path,
    input_hashes: Mapping[str, str],
) -> str | None:
    record = _read_step(store, step_id)
    if record is None:
        return None
    if record.get("status") != "completed":
        raise StateConflictError(f"saved step is not reusable: {step_id}")
    if record.get("input_hashes") != dict(sorted(input_hashes.items())):
        raise StateConflictError(f"saved step input hashes conflict: {step_id}")
    outputs = record.get("outputs")
    expected = outputs.get("sha256") if isinstance(outputs, Mapping) else None
    if not output_path.is_file() or not isinstance(expected, str):
        raise StateConflictError(f"saved step has no canonical output: {step_id}")
    if sha256_file(output_path) != expected:
        raise StateConflictError(f"saved output hash conflicts: {output_path}")
    return output_path.read_text(encoding="utf-8")


def _response_source(response: Mapping[str, object], step_id: str) -> str:
    source = response.get("composition_source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{step_id} response has no non-empty composition_source")
    return source.strip()


def _save_source(
    store: RunStore,
    step_id: str,
    output_path: Path,
    source: str,
    input_hashes: Mapping[str, str],
) -> Path:
    content = (source + "\n").encode("utf-8")
    digest = sha256_bytes(content)
    store.promote_bytes(content, output_path, digest)
    store.record_step(
        step_id,
        "completed",
        input_hashes,
        {"path": str(output_path.relative_to(store.run_dir)), "sha256": digest},
    )
    return output_path


def _material_ids(plan: PiecePlan) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for node in plan.nodes:
        material_id = node.score_material_id
        if material_id is not None and material_id not in seen:
            result.append(material_id)
            seen.add(material_id)
    return tuple(result)


def _identity_cues(plan: PiecePlan, material_ids: tuple[str, ...]) -> str:
    related = [
        {
            "node_id": node.node_id,
            "derived_from": node.derived_from,
            "score_material_id": node.score_material_id,
        }
        for node in plan.nodes
        if node.derived_from is not None and node.score_material_id in material_ids
    ]
    return _json_text(
        {
            "requested_material_ids": list(material_ids),
            "derived_occurrences": related,
            "assessment_scope": "same-material performance variation is automatically assessed",
        }
    )


def _score_summary(plan: PiecePlan, score: ScoreSpec) -> str:
    leaves, intervals = ordered_leaf_schedule(plan, score)
    materials = {material.material_id: material for material in score.materials}
    leaf_rows = []
    previous_harmony = None
    for node, _, _ in leaves:
        assert node.score_material_id is not None
        material = materials[node.score_material_id]
        first_harmony = material.harmonies[0] if material.harmonies else None
        last_harmony = material.harmonies[-1] if material.harmonies else None
        leaf_rows.append(
            {
                "node_id": node.node_id,
                "material_id": material.material_id,
                "has_harmony": bool(material.harmonies),
                "harmony_count": len(material.harmonies),
                "boundary_from_previous": None
                if previous_harmony is None or first_harmony is None
                else {
                    "previous_root_pitch_class": previous_harmony.root_pitch_class,
                    "previous_quality": previous_harmony.quality,
                    "next_root_pitch_class": first_harmony.root_pitch_class,
                    "next_quality": first_harmony.quality,
                },
            }
        )
        previous_harmony = last_harmony
    final_node = leaves[-1][0]
    assert final_node.score_material_id is not None
    final_material = materials[final_node.score_material_id]
    final_attack = max(note.at_units for note in final_material.notes)
    final_notes = tuple(note for note in final_material.notes if note.at_units == final_attack)
    return _json_text(
        {
            "score_id": score.score_id,
            "divisions": score.divisions,
            "total_occurrence_units": intervals[plan.root_node_id][1],
            "materials": [
                {
                    "material_id": material.material_id,
                    "length_units": material.length_units,
                    "note_count": len(material.notes),
                    "voices": sorted({note.voice for note in material.notes}),
                    "harmony_count": len(material.harmonies),
                    "harmonies": [
                        {
                            "at_units": harmony.at_units,
                            "duration_units": harmony.duration_units,
                            "root_pitch_class": harmony.root_pitch_class,
                            "quality": harmony.quality,
                        }
                        for harmony in material.harmonies
                    ],
                }
                for material in score.materials
            ],
            "leaves": leaf_rows,
            "ending": {
                "node_id": final_node.node_id,
                "material_id": final_material.material_id,
                "attack_units": final_attack,
                "pitch_classes": sorted({note.pitch % 12 for note in final_notes}),
                "minimum_duration_units": min(note.duration_units for note in final_notes),
            },
        }
    )


def _profile_catalog() -> str:
    return _json_text(
        {
            "catalog_version": "reference-conditioned-v2",
            "timing_budget_id": ["subtle-v1", "narrative-v1", "narrative-v2"],
            "timing_profile": ["neutral", "savor", "flow", "build", "release"],
            "timing_amount": ["subtle", "moderate"],
            "dynamics_profile": ["steady", "shape", "build", "release"],
            "articulation_profile": ["score", "legato", "light"],
            "coordination_profile": ["score", "rolled", "aligned"],
            "pedal_profile": {
                "selectable_values": ["none", "phrase_legato", "harmony_legato"],
                "inheritance": (
                    "each leaf uses the nearest explicit node setting; a root setting can "
                    "therefore affect the whole piece"
                ),
                "segment_rule": (
                    "contiguous leaves with the same owner and a non-harmony_legato value "
                    "are merged into one pedal segment"
                ),
                "profiles": {
                    "none": "explicitly produce no pedal events for the affected leaves",
                    "phrase_legato": (
                        "pedal once across each merged owner segment; it does not follow "
                        "local harmony changes"
                    ),
                    "harmony_legato": (
                        "requires declared score harmony and repedals at every local harmony "
                        "and material boundary"
                    ),
                },
            },
        }
    )


def _piece_plan(
    store: RunStore,
    runner: StageRunner,
    normalized_request: Mapping[str, object],
    prompt_target: Mapping[str, object],
    model_config: Mapping[str, object],
) -> tuple[PiecePlan, Path, str]:
    target = _stage_target(prompt_target, "piece_plan")
    prompt = build_stage_prompt(
        "piece_plan",
        {
            "REQUEST_JSON": _json_text(_anonymous_request(normalized_request)),
            "REFERENCE_TARGET_JSON": _json_text(target),
            "CALIBRATION_CONTRACT": _json_text(
                {
                    "preset": normalized_request.get("preset"),
                    "duration_ms": 180_000,
                    "requirements": [
                        "hierarchical form",
                        "declared contrast",
                        "varied recurrence",
                        "tonic ending",
                        (
                            "the final leaf uses a dedicated material with role=release "
                            "and no derived_from"
                        ),
                        (
                            "transition leaves are optional and never first or last in "
                            "performance order"
                        ),
                    ],
                }
            ),
        },
    )
    hashes = _input_hashes(prompt=prompt, target=target, model_config=model_config)
    path = store.run_dir / "outputs" / "piece-plan.dsl"
    saved = _reuse_source(store, "piece-plan", path, hashes)
    source = (
        saved.strip()
        if saved is not None
        else _response_source(runner.run("piece-plan", prompt, hashes), "piece-plan")
    )
    try:
        plan = parse_piece_plan(source)
        validate_piece_plan(plan)
    except (PipelineDslError, PipelineValidationError) as error:
        if saved is None:
            _save_stage_failure(store, "piece-plan", source, error, hashes)
        raise
    canonical = dump_piece_plan(plan)
    quality = evaluate_generic_piece_plan_quality(plan)
    tonal_targets = target.get("semantic_targets", [])
    tonal_target = (
        tonal_targets[0]
        if isinstance(tonal_targets, list) and tonal_targets
        else {"status": "unverified_continuous_reference"}
    )
    tonal_quality = evaluate_piece_plan_tonal_hierarchy(plan, tonal_target)
    quality = {
        **quality,
        "tonal_hierarchy": tonal_quality,
        "failures": [
            *quality.get("failures", []),
            *(f"tonal-hierarchy:{item}" for item in tonal_quality["failures"]),
        ],
    }
    quality["passes"] = not quality["failures"]
    quality_path = store.run_dir / "outputs" / "piece-plan-quality.json"
    quality_hashes = {
        "piece_plan": sha256_text(canonical),
        **staged_pipeline_fingerprints(),
    }
    _save_quality(store, "piece-plan-quality", quality_path, quality, quality_hashes)
    if not quality["passes"]:
        raise ValueError(f"piece plan quality failed: {quality['failures']}")
    if saved is None:
        _save_source(store, "piece-plan", path, canonical, hashes)
    return plan, path, canonical


def _score_spec(
    store: RunStore,
    runner: StageRunner,
    plan: PiecePlan,
    plan_source: str,
    prompt_target: Mapping[str, object],
    model_config: Mapping[str, object],
) -> tuple[ScoreSpec, Path, str]:
    expected_ids = _material_ids(plan)
    materials = []
    score_id: str | None = None
    divisions: int | None = None
    for start in range(0, len(expected_ids), MAX_MATERIALS_PER_BATCH):
        batch_ids = expected_ids[start : start + MAX_MATERIALS_PER_BATCH]
        number = start // MAX_MATERIALS_PER_BATCH + 1
        step_id = f"score-spec-{number:03d}"
        target = _stage_target(prompt_target, "score_spec")
        prior = ScoreSpec(score_id or "pending", divisions or 1, tuple(materials))
        prior_source = "[]" if not materials else dump_score_spec(prior)
        prompt = build_stage_prompt(
            "score_spec",
            {
                "PIECE_PLAN_DSL": plan_source,
                "REFERENCE_TARGET_JSON": _json_text(target),
                "SOURCE_MATERIALS_DSL": prior_source,
                "IDENTITY_CUES": _identity_cues(plan, batch_ids),
            },
        )
        hashes = _input_hashes(
            prompt=prompt,
            target=target,
            model_config=model_config,
            previous_source=plan_source + prior_source,
        )
        batch_path = store.run_dir / "outputs" / f"{step_id}.dsl"
        saved = _reuse_source(store, step_id, batch_path, hashes)
        source = (
            saved.strip()
            if saved is not None
            else _response_source(runner.run(step_id, prompt, hashes), step_id)
        )
        try:
            batch = parse_score_spec(source)
            canonical = dump_score_spec(batch)
            returned_ids = tuple(material.material_id for material in batch.materials)
            if returned_ids != batch_ids:
                raise StageResponseValidationError(
                    f"{step_id} returned material ids {returned_ids}, expected {batch_ids}"
                )
            if score_id is not None and batch.score_id != score_id:
                raise StageResponseValidationError(
                    "score batches must use the same score_id"
                )
            if divisions is not None and batch.divisions != divisions:
                raise StageResponseValidationError(
                    "score batches must use the same divisions"
                )
            partial_score = ScoreSpec(
                batch.score_id,
                batch.divisions,
                (*materials, *batch.materials),
            )
            validate_score_materials(partial_score)
        except (
            PipelineDslError,
            PipelineValidationError,
            StageResponseValidationError,
        ) as error:
            if saved is None:
                _save_stage_failure(store, step_id, source, error, hashes)
            raise
        if saved is None:
            _save_source(store, step_id, batch_path, canonical, hashes)
        score_id = partial_score.score_id
        divisions = partial_score.divisions
        materials = list(partial_score.materials)
    assert score_id is not None and divisions is not None
    score = ScoreSpec(score_id, divisions, tuple(materials))
    validate_score_spec(plan, score)
    source = dump_score_spec(score)
    quality_path = store.run_dir / "outputs" / "score-quality.json"
    quality_hashes = _quality_input_hashes(
        plan_source=plan_source,
        score_source=source,
    )
    quality = evaluate_generic_score_quality(plan, score)
    _save_quality(store, "score-quality", quality_path, quality, quality_hashes)
    if not quality["passes"]:
        raise ValueError(f"score quality failed: {quality['failures']}")
    output = store.run_dir / "outputs" / "score-spec.dsl"
    aggregate_hashes = {
        "piece_plan": sha256_text(plan_source),
        "batches": sha256_text(source),
        **staged_pipeline_fingerprints(),
    }
    saved = _reuse_source(store, "score-spec-aggregate", output, aggregate_hashes)
    if saved is None:
        _save_source(store, "score-spec-aggregate", output, source, aggregate_hashes)
    elif saved.strip() != source:
        raise StateConflictError("saved score-spec aggregate conflicts with score batches")
    return score, output, source


def _performance_spec(
    store: RunStore,
    runner: StageRunner,
    plan: PiecePlan,
    plan_source: str,
    score: ScoreSpec,
    score_source: str,
    prompt_target: Mapping[str, object],
    model_config: Mapping[str, object],
) -> tuple[PerformanceSpec, Path, str, RenderedPerformance, dict[str, Any]]:
    target = _stage_target(prompt_target, "performance_spec")
    prompt = build_stage_prompt(
        "performance_spec",
        {
            "PIECE_PLAN_DSL": plan_source,
            "REFERENCE_TARGET_JSON": _json_text(target),
            "SCORE_SUMMARY": _score_summary(plan, score),
            "PROFILE_CATALOG": _profile_catalog(),
        },
    )
    hashes = _input_hashes(
        prompt=prompt,
        target=target,
        model_config=model_config,
        previous_source=plan_source + score_source,
    )
    path = store.run_dir / "outputs" / "performance-spec.dsl"
    saved = _reuse_source(store, "performance-spec", path, hashes)
    source = (
        saved.strip()
        if saved is not None
        else _response_source(runner.run("performance-spec", prompt, hashes), "performance-spec")
    )
    try:
        performance = parse_performance_spec(source)
        validate_pipeline(plan, score, performance)
    except (PipelineDslError, PipelineValidationError) as error:
        if saved is None:
            _save_stage_failure(store, "performance-spec", source, error, hashes)
        raise
    canonical = dump_performance_spec(performance)
    quality_path = store.run_dir / "outputs" / "performance-quality.json"
    quality_hashes = _quality_input_hashes(
        plan_source=plan_source,
        score_source=score_source,
        performance_source=canonical,
    )
    try:
        rendered = render_performance(plan, score, performance)
        quality = evaluate_generic_pipeline_quality(plan, score, performance, rendered)
    except Exception as error:
        quality = {
            "schema_version": 1,
            "passes": False,
            "failures": ["performance-quality-error"],
            "error_type": type(error).__name__,
            "detail": str(error),
        }
        _save_quality(store, "performance-quality", quality_path, quality, quality_hashes)
        raise ValueError(f"performance quality failed: {error}") from error
    _save_quality(store, "performance-quality", quality_path, quality, quality_hashes)
    if not quality["passes"]:
        raise ValueError(f"generic quality gate failed: {quality['failures']}")
    if saved is None:
        _save_source(store, "performance-spec", path, canonical, hashes)
    return performance, path, canonical, rendered, quality


def _publish(
    store: RunStore,
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
    rendered: RenderedPerformance,
    quality: dict[str, Any],
    sources: tuple[str, str, str],
) -> tuple[Path, Path, dict[str, Any]]:
    musicxml_path = store.run_dir / "outputs" / "score.musicxml"
    smf_path = store.run_dir / "outputs" / "final.mid"
    quality_path = store.run_dir / "outputs" / "quality.json"
    saved = _read_step(store, "publish-final")
    if saved is not None:
        outputs = saved.get("outputs")
        if saved.get("status") != "completed" or not isinstance(outputs, Mapping):
            raise StateConflictError("saved publish-final step is not reusable")
        checks = (
            (musicxml_path, outputs.get("musicxml_sha256")),
            (smf_path, outputs.get("smf_sha256")),
            (quality_path, outputs.get("quality_sha256")),
        )
        if any(
            not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected
            for path, expected in checks
        ):
            raise StateConflictError("saved publish-final output hash conflicts")
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        if not isinstance(quality, dict):
            raise StateConflictError("saved quality result is not an object")
        return musicxml_path, smf_path, quality

    store.snapshot_json(quality_path.relative_to(store.run_dir), quality)
    musicxml_path = render_musicxml(plan, score, musicxml_path)
    ET.parse(musicxml_path)
    render_performance_smf(rendered, smf_path)
    if smf_path.read_bytes()[:4] != b"MThd":
        raise ValueError("rendered SMF has no MThd header")
    expected_notes = sorted(
        (note.pitch, note.at_ms, note.duration_ms, note.velocity) for note in rendered.notes
    )
    actual_notes = sorted(
        (note.pitch, note.onset_ms, note.duration_ms, note.velocity)
        for note in load_smf_notes(smf_path)
    )
    if actual_notes != expected_notes:
        raise ValueError("rendered SMF note round-trip does not match the performance")
    hashes = {
        "piece_plan": sha256_text(sources[0]),
        "score_spec": sha256_text(sources[1]),
        "performance_spec": sha256_text(sources[2]),
        **staged_pipeline_fingerprints(),
    }
    store.record_step(
        "publish-final",
        "completed",
        hashes,
        {
            "musicxml_path": str(musicxml_path.relative_to(store.run_dir)),
            "musicxml_sha256": sha256_file(musicxml_path),
            "smf_path": str(smf_path.relative_to(store.run_dir)),
            "smf_sha256": sha256_file(smf_path),
            "quality_path": str(quality_path.relative_to(store.run_dir)),
            "quality_sha256": sha256_file(quality_path),
        },
    )
    return musicxml_path, smf_path, quality


def run_staged_pipeline_generation(
    *,
    run_dir: Path,
    normalized_request: Mapping[str, object],
    resolved_request: Mapping[str, object],
    prompt_target: Mapping[str, object],
    model_config: Mapping[str, object],
    runner: StageRunner,
    stop_after: StopAfter,
) -> StagedPipelineResult:
    """確定済み段階を再利用しながら、指定段階まで生成する。"""

    if stop_after not in {"piece_plan", "score_spec", "performance_spec"}:
        raise ValueError(f"unknown stop_after: {stop_after}")
    store = RunStore(run_dir)
    spec = {
        "schema_version": 1,
        "pipeline": "reference-conditioned-staged-v1",
        "normalized_request_sha256": sha256_json(normalized_request),
        "resolved_request_sha256": sha256_json(resolved_request),
        "prompt_target_sha256": sha256_json(prompt_target),
        "model_config_sha256": sha256_json(model_config),
        "implementation_hashes": staged_pipeline_fingerprints(),
    }
    with RunLock(Path(run_dir) / ".run.lock"):
        store.initialize(spec)
        _validate_failure_state(store)
        _validate_quality_state(store)
        store.snapshot_json("inputs/normalized-request.json", normalized_request)
        store.snapshot_json("inputs/resolved-request.json", resolved_request)
        store.snapshot_json("inputs/prompt-target.json", prompt_target)
        store.snapshot_json("inputs/model-config.json", model_config)
        plan, plan_path, plan_source = _piece_plan(
            store, runner, normalized_request, prompt_target, model_config
        )
        if stop_after == "piece_plan":
            return StagedPipelineResult(
                "stopped", "piece_plan", plan_path, None, None, None, None, None
            )
        score, score_path, score_source = _score_spec(
            store, runner, plan, plan_source, prompt_target, model_config
        )
        if stop_after == "score_spec":
            return StagedPipelineResult(
                "stopped", "score_spec", plan_path, score_path, None, None, None, None
            )
        performance, performance_path, performance_source, rendered, quality = _performance_spec(
            store,
            runner,
            plan,
            plan_source,
            score,
            score_source,
            prompt_target,
            model_config,
        )
        musicxml, smf, quality = _publish(
            store,
            plan,
            score,
            performance,
            rendered,
            quality,
            (plan_source, score_source, performance_source),
        )
        return StagedPipelineResult(
            "completed",
            None,
            plan_path,
            score_path,
            performance_path,
            musicxml,
            smf,
            quality,
        )
