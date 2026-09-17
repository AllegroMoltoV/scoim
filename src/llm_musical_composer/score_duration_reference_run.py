"""同一ScoreSpecの音価だけを二つの匿名参照方向へ変更して評価する。"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from llm_musical_composer.generic_pipeline_quality import (
    evaluate_generic_pipeline_quality,
    evaluate_generic_score_quality,
)
from llm_musical_composer.performance_pipeline import (
    PerformanceSpec,
    PiecePlan,
    RenderedPerformance,
    ScoreSpec,
    render_musicxml,
    render_performance,
    render_performance_smf,
    validate_score_spec,
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
from llm_musical_composer.reference_profile import (
    ReferencePiece,
    build_copy_fingerprint,
    evaluate_copy_risk,
    extract_reference_profile,
    load_reference_piece,
)
from llm_musical_composer.run_state import (
    RunStore,
    atomic_write_json,
    sha256_file,
    sha256_json,
    sha256_text,
)
from llm_musical_composer.whole_score_staged_generation_run import (
    WholeScoreLiveRunError,
    calibrate_key_release,
)

PROTOCOL_ID = "score-duration-reference-pair-v1"
MODEL_CONFIG = {
    "model": "gpt-5.6-sol",
    "reasoning_effort": "high",
    "timeout_seconds": 900,
    "maximum_external_calls": 2,
}
BASE_ROOT = Path(
    ".appendix/key-release-calibration-v1/reference-v7-a2-joint-texture-v4"
)
TARGETS = (
    (
        "candidate-a",
        Path(
            ".appendix/whole-score-staged-generation-live-v4/"
            "reference-v7-a2-joint-texture-v4/inputs/prompt-target.json"
        ),
    ),
    (
        "candidate-b",
        Path(
            ".appendix/key-release-rhythm-feasibility-v1/inputs/"
            "under-the-cold-sky-target-v3.json"
        ),
    ),
)
EXPECTED_HASHES = {
    "piece_plan": "2bf344b0cc448dea2ec604de071b7ff9ed3047fc71cf1f86c64100e7c3eba419",
    "score_spec": "f89622d3eed135613d6df762b9953a0893922f270d50e461ce98e7eb24eb8f48",
    "performance_spec": "89c1dc3ba9f6635edd8dcd52d0bf2f0f68d256e5e13f122f3e8680623bf1a213",
    "target_a": "b7f3575b0edfff704ccfca30b6b8420a316c1d58f48bc166d822821e0c8d8ec4",
    "target_b": "4549b609273b1a2d3045a8208b4d1118aaab2230eecdd9431d58989910b912b1",
}


class ScoreDurationReferenceError(ValueError):
    """音価参照実験の入力または応答が契約外の場合に送出する。"""


class DurationRunner(Protocol):
    @property
    def call_number(self) -> int: ...

    def run(
        self,
        step_id: str,
        prompt: str,
        input_hashes: dict[str, str] | None = None,
    ) -> dict[str, object]: ...


@dataclass(frozen=True)
class DurationEdit:
    event_id: str
    duration_units: int


def _call(node: ast.AST, name: str) -> ast.Call:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ScoreDurationReferenceError(f"expected {name} call")
    if node.func.id != name or node.args:
        raise ScoreDurationReferenceError(f"expected {name} keyword-only call")
    return node


def _keywords(call: ast.Call, required: set[str]) -> dict[str, ast.AST]:
    if any(item.arg is None for item in call.keywords):
        raise ScoreDurationReferenceError("expanded keyword arguments are not supported")
    values = {str(item.arg): item.value for item in call.keywords}
    if len(values) != len(call.keywords) or set(values) != required:
        raise ScoreDurationReferenceError("duration patch fields are invalid")
    return values


def parse_score_duration_patch(source: str) -> tuple[DurationEdit, ...]:
    """音符IDと正整数音価だけを持つ一時DSLを解析する。"""

    try:
        expression = ast.parse(source.strip(), mode="eval").body
    except (SyntaxError, ValueError) as error:
        raise ScoreDurationReferenceError(f"duration patch syntax is invalid: {error}") from error
    root = _keywords(_call(expression, "score_duration_patch"), {"edits"})
    raw_edits = root["edits"]
    if not isinstance(raw_edits, (ast.List, ast.Tuple)):
        raise ScoreDurationReferenceError("duration patch edits must be a list")
    edits: list[DurationEdit] = []
    for node in raw_edits.elts:
        values = _keywords(
            _call(node, "duration_edit"),
            {"event_id", "duration_units"},
        )
        try:
            event_id = ast.literal_eval(values["event_id"])
            duration_units = ast.literal_eval(values["duration_units"])
        except (ValueError, TypeError) as error:
            raise ScoreDurationReferenceError("duration edit literals are invalid") from error
        if not isinstance(event_id, str) or not event_id:
            raise ScoreDurationReferenceError("event_id must be a non-empty string")
        if isinstance(duration_units, bool) or not isinstance(duration_units, int):
            raise ScoreDurationReferenceError("duration_units must be an integer")
        if duration_units <= 0:
            raise ScoreDurationReferenceError("duration_units must be positive")
        edits.append(DurationEdit(event_id, duration_units))
    if not edits:
        raise ScoreDurationReferenceError("duration patch needs at least one edit")
    ids = [item.event_id for item in edits]
    if len(ids) != len(set(ids)):
        raise ScoreDurationReferenceError("duration edit event IDs are duplicated")
    return tuple(edits)


def _ordered_leaves(plan: PiecePlan):
    nodes = {item.node_id: item for item in plan.nodes}
    children = {item.node_id: [] for item in plan.nodes}
    for node in plan.nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node)

    def visit(node):
        descendants = sorted(children[node.node_id], key=lambda item: item.order)
        if not descendants:
            return (node,)
        return tuple(leaf for child in descendants for leaf in visit(child))

    return visit(nodes[plan.root_node_id])


def material_context(plan: PiecePlan, score: ScoreSpec) -> list[dict[str, Any]]:
    """固有素材ごとの全曲出現数と終止固定状態を返す。"""

    leaves = _ordered_leaves(plan)
    counts: dict[str, int] = {}
    releases: set[str] = set()
    for leaf in leaves:
        assert leaf.score_material_id is not None
        counts[leaf.score_material_id] = counts.get(leaf.score_material_id, 0) + 1
        if leaf.role == "release":
            releases.add(leaf.score_material_id)
    return [
        {
            "material_id": material.material_id,
            "length_units": material.length_units,
            "occurrence_count": counts.get(material.material_id, 0),
            "release_fixed": material.material_id in releases,
        }
        for material in score.materials
    ]


def count_cross_voice_pitch_overlaps(score: ScoreSpec) -> int:
    """素材内で上下声が同じ物理鍵を同時保持する組を数える。"""

    count = 0
    for material in score.materials:
        by_pitch: dict[int, dict[str, list[Any]]] = {}
        for note in material.notes:
            by_pitch.setdefault(note.pitch, {"upper": [], "lower": []})[note.voice].append(note)
        for voices in by_pitch.values():
            for upper in voices["upper"]:
                for lower in voices["lower"]:
                    if max(upper.at_units, lower.at_units) < min(
                        upper.at_units + upper.duration_units,
                        lower.at_units + lower.duration_units,
                    ):
                        count += 1
    return count


def apply_score_duration_patch(
    plan: PiecePlan,
    score: ScoreSpec,
    edits: tuple[DurationEdit, ...],
) -> ScoreSpec:
    """検証済みの音価変更だけを基準ScoreSpecへ決定的に適用する。"""

    if not edits:
        raise ScoreDurationReferenceError("duration patch needs at least one edit")
    edit_by_id = {item.event_id: item.duration_units for item in edits}
    if len(edit_by_id) != len(edits):
        raise ScoreDurationReferenceError("duration edit event IDs are duplicated")
    context = {item["material_id"]: item for item in material_context(plan, score)}
    known: dict[str, tuple[str, Any]] = {
        note.event_id: (material.material_id, note)
        for material in score.materials
        for note in material.notes
    }
    unknown = set(edit_by_id) - set(known)
    if unknown:
        raise ScoreDurationReferenceError(
            f"duration patch contains unknown event ID: {min(unknown)}"
        )
    for event_id, duration in edit_by_id.items():
        material_id, note = known[event_id]
        if context[material_id]["release_fixed"]:
            raise ScoreDurationReferenceError("release material duration is fixed")
        if note.at_units + duration > context[material_id]["length_units"]:
            raise ScoreDurationReferenceError("duration edit extends outside its material")
    baseline_overlap = count_cross_voice_pitch_overlaps(score)
    materials = tuple(
        replace(
            material,
            notes=tuple(
                replace(note, duration_units=edit_by_id[note.event_id])
                if note.event_id in edit_by_id
                else note
                for note in material.notes
            ),
        )
        for material in score.materials
    )
    changed = replace(score, materials=materials)
    try:
        validate_score_spec(plan, changed)
    except ValueError as error:
        raise ScoreDurationReferenceError(f"duration patch violates ScoreSpec: {error}") from error
    if count_cross_voice_pitch_overlaps(changed) > baseline_overlap:
        raise ScoreDurationReferenceError("duration patch adds cross-voice same-pitch overlap")
    return changed


def _rendered_piece(rendered: RenderedPerformance) -> ReferencePiece:
    return ReferencePiece.from_dicts(
        name="anonymous-generated-performance",
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


def rendered_duration_distribution(
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
) -> list[float]:
    """キー解放率100%でSMFと同じduration_ratioを測る。"""

    rendered = render_performance(
        plan,
        score,
        replace(performance, key_release_percent=100),
    )
    profile = extract_reference_profile(_rendered_piece(rendered))
    return [
        float(value)
        for value in profile["feature_groups"]["rhythm_time"]["metrics"][
            "duration_ratio"
        ]["values"]
    ]


def _duration_target(target: dict[str, Any]) -> dict[str, Any]:
    try:
        matches = [
            item
            for item in target["stage_targets"]["rendered_surface"]
            if item["id"] == "rhythm_time.duration_ratio"
        ]
    except (KeyError, TypeError) as error:
        raise ScoreDurationReferenceError("duration target is missing") from error
    if len(matches) != 1:
        raise ScoreDurationReferenceError("duration target is missing or duplicated")
    return matches[0]


def build_duration_target_view(
    target: dict[str, Any],
    baseline_at_100: list[float],
) -> dict[str, Any]:
    """低水準の六区分を匿名の意味付きviewへ変換する。"""

    center = [float(value) for value in target["neighborhood_center"]]
    if len(center) != 6 or len(baseline_at_100) != 6:
        raise ScoreDurationReferenceError("duration target must have six bins")
    bins = (
        ("very_short", "全曲中央値発音間隔の0.5倍以下"),
        ("short", "全曲中央値発音間隔の0.5倍より長く約0.707倍以下"),
        ("near_one_short", "全曲中央値発音間隔の約0.707倍より長く1倍以下"),
        ("near_one_long", "全曲中央値発音間隔の1倍より長く約1.414倍以下"),
        ("long", "全曲中央値発音間隔の約1.414倍より長く2倍以下"),
        ("very_long", "全曲中央値発音間隔の2倍より長い"),
    )
    baseline = [float(value) for value in baseline_at_100]
    return {
        "schema_version": 1,
        "descriptor_id": "rhythm_time.duration_ratio",
        "kind": "distribution",
        "bins": [{"id": item[0], "meaning": item[1]} for item in bins],
        "neighborhood_center": center,
        "neighborhood_radius": float(target["neighborhood_radius"]),
        "baseline_at_key_release_100": baseline,
        "residual_from_center": [
            actual - expected for actual, expected in zip(baseline, center, strict=True)
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ScoreDurationReferenceError(f"JSON root must be an object: {path}")
    return value


def _save_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _distance(actual: list[float], center: list[float]) -> float:
    return sum(abs(left - right) for left, right in zip(actual, center, strict=True)) / 2


def _projection(actual: list[float], target_a: list[float], target_b: list[float]) -> float:
    axis = [left - right for left, right in zip(target_a, target_b, strict=True)]
    denominator = sum(value * value for value in axis)
    if denominator <= 0:
        raise ScoreDurationReferenceError("reference duration centers are identical")
    return sum(
        (value - origin) * direction
        for value, origin, direction in zip(actual, target_b, axis, strict=True)
    ) / denominator


def _invariants(rendered: RenderedPerformance) -> dict[str, Any]:
    notes = [
        (note.at_ms, note.pitch, note.velocity, note.voice)
        for note in rendered.notes
    ]
    pedals = [(item.at_ms, item.value) for item in rendered.pedals]
    attacks = sorted({item[0] for item in notes})
    return {
        "note_count": len(notes),
        "attack_count": len(attacks),
        "attack_pitch_velocity_voice_sha256": sha256_json(notes),
        "pedal_sha256": sha256_json(pedals),
        "terminal_notes": [
            asdict(note)
            for note in rendered.notes
            if note.at_ms == max(item.at_ms for item in rendered.notes)
        ],
    }


def _copy_risk(project_root: Path, piece: ReferencePiece) -> dict[str, Any]:
    records_path = project_root / ".appendix/reference-profile-v1/files.jsonl"
    references: dict[str, dict[str, Any]] = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if isinstance(record.get("name"), str) and isinstance(
            record.get("copy_fingerprint"), dict
        ):
            references[record["name"]] = record["copy_fingerprint"]
    return evaluate_copy_risk(
        build_copy_fingerprint(piece),
        references,
        review_threshold=0.8,
    )


def _prompt(
    template: str,
    score_source: str,
    context: list[dict[str, Any]],
    target_view: dict[str, Any],
) -> str:
    public_context = {
        "material_context": context,
        "target": target_view,
        "rules": {
            "weighted_by_occurrence_count": True,
            "release_material_is_fixed": True,
            "preserve_all_non_duration_fields": True,
        },
    }
    return template.replace(
        "{{CONTEXT_JSON}}",
        json.dumps(public_context, ensure_ascii=False, indent=2, sort_keys=True),
    ).replace("{{SCORE_SPEC}}", score_source)


def _verify_fixed_inputs(project_root: Path) -> dict[str, Path]:
    paths = {
        "piece_plan": project_root / BASE_ROOT / "inputs/piece-plan.dsl",
        "score_spec": project_root / BASE_ROOT / "outputs/score-spec.dsl",
        "performance_spec": project_root / BASE_ROOT / "outputs/performance-spec.dsl",
        "target_a": project_root / TARGETS[0][1],
        "target_b": project_root / TARGETS[1][1],
    }
    for key, expected in EXPECTED_HASHES.items():
        actual = sha256_file(paths[key])
        if actual != expected:
            raise ScoreDurationReferenceError(f"fixed input hash differs: {key}")
    return paths


def _candidate(
    project_root: Path,
    run_root: Path,
    runner: DurationRunner,
    step_id: str,
    target_path: Path,
    plan: PiecePlan,
    score: ScoreSpec,
    performance: PerformanceSpec,
    baseline_distribution: list[float],
    baseline_invariants: dict[str, Any],
    template: str,
) -> dict[str, Any]:
    candidate_dir = run_root / step_id
    target = _read_json(target_path)
    view = build_duration_target_view(_duration_target(target), baseline_distribution)
    context = material_context(plan, score)
    score_source = dump_score_spec(score)
    prompt = _prompt(template, score_source, context, view)
    _save_text(candidate_dir / "inputs/base-score-spec.dsl", score_source)
    atomic_write_json(candidate_dir / "inputs/material-context.json", context)
    atomic_write_json(candidate_dir / "inputs/duration-target-view.json", view)
    _save_text(candidate_dir / "prompts/score-duration.md", prompt)
    input_hashes = {
        "score_spec": sha256_text(score_source),
        "material_context": sha256_json(context),
        "target_view": sha256_json(view),
        "prompt": sha256_text(prompt),
    }
    try:
        response = runner.run(step_id, prompt, input_hashes)
        source = response.get("composition_source")
        if not isinstance(source, str) or not source.strip():
            raise ScoreDurationReferenceError("response has no composition_source")
        _save_text(candidate_dir / "responses/duration-patch.dsl", source)
        patch = parse_score_duration_patch(source)
        changed = apply_score_duration_patch(plan, score, patch)
        changed_source = dump_score_spec(changed)
        _save_text(candidate_dir / "outputs/score-spec.dsl", changed_source)

        common_performance = replace(performance, key_release_percent=100)
        rendered = render_performance(plan, changed, common_performance)
        score_quality = evaluate_generic_score_quality(plan, changed)
        pipeline_quality = evaluate_generic_pipeline_quality(
            plan, changed, common_performance, rendered
        )
        duration = rendered_duration_distribution(plan, changed, common_performance)
        invariants = _invariants(rendered)
        invariant_checks = {
            "attack_count": invariants["attack_count"] == baseline_invariants["attack_count"],
            "attack_pitch_velocity_voice": invariants[
                "attack_pitch_velocity_voice_sha256"
            ]
            == baseline_invariants["attack_pitch_velocity_voice_sha256"],
            "pedal": invariants["pedal_sha256"] == baseline_invariants["pedal_sha256"],
            "terminal": invariants["terminal_notes"] == baseline_invariants["terminal_notes"],
            "cross_voice_same_pitch_overlap": count_cross_voice_pitch_overlaps(changed)
            <= count_cross_voice_pitch_overlaps(score),
        }
        common_dir = candidate_dir / "outputs/common-100"
        common_musicxml = render_musicxml(plan, changed, common_dir / "final.musicxml")
        common_smf = render_performance_smf(rendered, common_dir / "final.mid").path
        reloaded = extract_reference_profile(load_reference_piece(common_smf))[
            "feature_groups"
        ]["rhythm_time"]["metrics"]["duration_ratio"]["values"]
        memory_smf_equal = duration == [float(value) for value in reloaded]
        copy_risk = _copy_risk(project_root, load_reference_piece(common_smf))

        key_target = next(
            item
            for item in target["semantic_targets"]["rendered_surface"]
            if item["id"] == "key_held_texture"
        )
        calibration: dict[str, Any]
        final_smf: str | None = None
        try:
            calibrated_performance, calibrated_rendered, calibration = calibrate_key_release(
                plan, changed, performance, key_target, minimum_percent=40
            )
            calibrated_dir = candidate_dir / "outputs/calibrated"
            _save_text(
                calibrated_dir / "performance-spec.dsl",
                dump_performance_spec(calibrated_performance),
            )
            render_musicxml(plan, changed, calibrated_dir / "final.musicxml")
            final_path = render_performance_smf(
                calibrated_rendered, calibrated_dir / "final.mid"
            ).path
            final_smf = _display_path(project_root, final_path)
        except WholeScoreLiveRunError as error:
            calibration = {"status": "unreachable", "detail": str(error)}

        result = {
            "status": "completed",
            "step_id": step_id,
            "edit_count": len(patch),
            "duration_distribution_at_100": duration,
            "distance_to_own_center": _distance(duration, view["neighborhood_center"]),
            "within_own_neighborhood": _distance(duration, view["neighborhood_center"])
            <= view["neighborhood_radius"],
            "score_quality": score_quality,
            "pipeline_quality": pipeline_quality,
            "invariants": invariants,
            "invariant_checks": invariant_checks,
            "memory_smf_duration_equal": memory_smf_equal,
            "copy_risk": copy_risk,
            "key_release_calibration": calibration,
            "common_smf": _display_path(project_root, common_smf),
            "common_musicxml": _display_path(project_root, common_musicxml),
            "final_smf": final_smf,
        }
        atomic_write_json(candidate_dir / "result.json", result)
        return result
    except Exception as error:
        result = {
            "status": "failed",
            "step_id": step_id,
            "error": {"type": type(error).__name__, "detail": str(error)},
        }
        atomic_write_json(candidate_dir / "result.json", result)
        return result


def run_reference_pair(
    project_root: Path,
    run_root: Path,
    runner: DurationRunner | None = None,
) -> dict[str, Any]:
    """二つの匿名参照方向を各一回だけ実走し、方向応答を判定する。"""

    project_root = Path(project_root).resolve()
    run_root = Path(run_root)
    if not run_root.is_absolute():
        run_root = project_root / run_root
    run_root = run_root.resolve()
    paths = _verify_fixed_inputs(project_root)
    plan = parse_piece_plan(paths["piece_plan"].read_text(encoding="utf-8"))
    score = parse_score_spec(paths["score_spec"].read_text(encoding="utf-8"))
    performance = parse_performance_spec(
        paths["performance_spec"].read_text(encoding="utf-8")
    )
    baseline_performance = replace(performance, key_release_percent=100)
    baseline_rendered = render_performance(plan, score, baseline_performance)
    baseline_distribution = rendered_duration_distribution(plan, score, performance)
    baseline_invariants = _invariants(baseline_rendered)
    target_values = [
        build_duration_target_view(
            _duration_target(_read_json(paths[f"target_{suffix}"])),
            baseline_distribution,
        )["neighborhood_center"]
        for suffix in ("a", "b")
    ]
    spec = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model": MODEL_CONFIG,
        "fixed_input_hashes": EXPECTED_HASHES,
        "candidate_order": [item[0] for item in TARGETS],
    }
    store = RunStore(run_root, max_calls=MODEL_CONFIG["maximum_external_calls"])
    store.initialize(spec)
    atomic_write_json(
        run_root / "baseline.json",
        {
            "duration_distribution_at_100": baseline_distribution,
            "invariants": baseline_invariants,
            "cross_voice_same_pitch_overlap_count": count_cross_voice_pitch_overlaps(score),
        },
    )
    template = (project_root / "prompts/pipeline-score-duration-reference.md").read_text(
        encoding="utf-8"
    )
    active_runner = runner or CodexExecRunner(
        run_store=store,
        schema_path=project_root / "schemas/codex-composition-response.schema.json",
        model=MODEL_CONFIG["model"],
        reasoning_effort=MODEL_CONFIG["reasoning_effort"],
        working_directory=isolated_codex_working_directory(),
        timeout_seconds=MODEL_CONFIG["timeout_seconds"],
    )
    results = []
    for (step_id, _), key in zip(TARGETS, ("target_a", "target_b"), strict=True):
        results.append(
            _candidate(
                project_root,
                run_root,
                active_runner,
                step_id,
                paths[key],
                plan,
                score,
                performance,
                baseline_distribution,
                baseline_invariants,
                template,
            )
        )
    baseline_projection = _projection(
        baseline_distribution, target_values[0], target_values[1]
    )
    projections: dict[str, float | None] = {"baseline": baseline_projection}
    for result in results:
        projections[result["step_id"]] = (
            _projection(
                result["duration_distribution_at_100"],
                target_values[0],
                target_values[1],
            )
            if result["status"] == "completed"
            else None
        )
    mechanical_pass = False
    if all(item["status"] == "completed" for item in results):
        a, b = results
        mechanical_pass = bool(
            projections["candidate-a"] > baseline_projection
            and projections["candidate-b"] < baseline_projection
            and projections["candidate-a"] > projections["candidate-b"]
            and all(a["invariant_checks"].values())
            and all(b["invariant_checks"].values())
            and a["score_quality"]["passes"]
            and a["pipeline_quality"]["passes"]
            and b["score_quality"]["passes"]
            and b["pipeline_quality"]["passes"]
            and a["memory_smf_duration_equal"]
            and b["memory_smf_duration_equal"]
            and not a["copy_risk"]["exact"]
            and not b["copy_risk"]["exact"]
            and a["key_release_calibration"]["status"] == "pass"
            and b["key_release_calibration"]["status"] == "pass"
        )
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "confirmed_external_call_count": active_runner.call_number,
        "mechanical_two_way_response": mechanical_pass,
        "baseline_duration_distribution_at_100": baseline_distribution,
        "target_centers": {"candidate-a": target_values[0], "candidate-b": target_values[1]},
        "projections": projections,
        "candidates": results,
        "human_review_required": mechanical_pass,
    }
    atomic_write_json(run_root / "result.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_reference_pair(arguments.project_root, arguments.run_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
