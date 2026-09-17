"""無応答になった通常生成の終止を決定的に完走する。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from llm_musical_composer.generic_pipeline_quality import (
    evaluate_generic_pipeline_quality,
    evaluate_generic_score_quality,
)
from llm_musical_composer.harmonic_skeleton import HarmonicSkeletonV0
from llm_musical_composer.performance_pipeline import (
    PiecePlan,
    ScoreSpec,
    render_musicxml,
    render_performance_smf,
)
from llm_musical_composer.pipeline_dsl import (
    dump_performance_spec,
    dump_score_spec,
    parse_piece_plan,
)
from llm_musical_composer.run_state import (
    RunStore,
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    sha256_json,
    sha256_text,
)
from llm_musical_composer.staged_material_pilot import TextureDraft
from llm_musical_composer.supported_normal_generation_repair import (
    V5_BATCH_3_CANONICAL_SHA256,
    V5_BATCH_3_RAW_SHA256,
    V6_HARMONY_SHA256,
    V6_MELODY_SHA256,
    V6_TEXTURE_BUDGET_SHA256,
    V7_PREFIX_SHA256,
    prepare_supported_normal_generation_repair,
)
from llm_musical_composer.texture_budget import (
    measure_rendered_texture_budget,
    measure_score_texture_budget,
)
from llm_musical_composer.whole_score_staged_generation import (
    PerformanceOccurrenceDraftV0,
    ScorePayloadV0,
    WholeScoreTextureResultV0,
    assemble_whole_score_performance,
    assemble_whole_score_textures,
)
from llm_musical_composer.whole_score_staged_generation_dsl import (
    dump_performance_collection,
    dump_texture_collection,
    parse_texture_collection,
)
from llm_musical_composer.whole_score_staged_generation_run import (
    calibrate_key_release,
    evaluate_staged_candidate,
    execute_harmony_stage,
    execute_melody_stages,
)

PROTOCOL_ID = "supported-normal-generation-v8-deterministic-terminal-fallback"
DEFAULT_ARTIFACT_ROOT = Path(
    ".appendix/supported-normal-generation-v8-deterministic-terminal-fallback"
)
V5_ROOT = Path(".appendix/supported-normal-generation-v5-placement-backtracking")
V7_ROOT = Path(".appendix/supported-normal-generation-v7-final-release-only")
SOURCE_FINAL_ONSET_UNITS = 52
TARGET_FINAL_ONSET_UNITS = 59


class SupportedNormalGenerationTerminalFallbackError(ValueError):
    """終端fallbackの固定来歴または品質契約に違反した。"""


class _NoExternalRunner:
    @property
    def call_number(self) -> int:
        return 0

    def run(self, step_id: str, prompt: str, input_hashes=None):
        del prompt, input_hashes
        raise SupportedNormalGenerationTerminalFallbackError(
            f"external call is forbidden in terminal fallback: {step_id}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupportedNormalGenerationTerminalFallbackError(
            f"JSON object required: {path}"
        )
    return value


def _write_text(path: Path, source: str) -> None:
    atomic_write_bytes(path, source.encode("utf-8"))


def _final_material_budget(texture_budget: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for item in texture_budget.get("materials", [])
        if item.get("material_id") == "material_final_release"
    ]
    if len(matches) != 1:
        raise SupportedNormalGenerationTerminalFallbackError(
            "final release texture budget is missing"
        )
    return matches[0]


def _initial_terminal_events(source_final: TextureDraft):
    fifth_indexes = [
        index
        for index, event in enumerate(source_final.events)
        if event.at_units == SOURCE_FINAL_ONSET_UNITS and event.degree == "fifth"
    ]
    root_indexes = [
        index
        for index, event in enumerate(source_final.events)
        if event.at_units == TARGET_FINAL_ONSET_UNITS and event.degree == "root"
    ]
    if len(fifth_indexes) != 1 or len(root_indexes) != 1:
        raise SupportedNormalGenerationTerminalFallbackError(
            "fixed source ending texture does not have the expected fifth and root"
        )
    fifth_index = fifth_indexes[0]
    root_index = root_indexes[0]
    events = []
    for index, event in enumerate(source_final.events):
        if index == fifth_index:
            events.append(
                replace(
                    event,
                    at_units=TARGET_FINAL_ONSET_UNITS,
                    duration_units=8,
                )
            )
        elif index == root_index:
            events.append(replace(event, duration_units=8))
        elif (
            event.at_units < TARGET_FINAL_ONSET_UNITS
            < event.at_units + event.duration_units
        ):
            events.append(
                replace(
                    event,
                    duration_units=TARGET_FINAL_ONSET_UNITS - event.at_units,
                )
            )
        else:
            events.append(event)
    return tuple(events)


def _combined_group_sizes(final_melody, events) -> Counter[int]:
    sizes = Counter(note.at_units for note in final_melody.notes)
    sizes.update(event.at_units for event in events)
    return sizes


def _assemble_terminal_candidate(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    melodies: ScorePayloadV0,
    prefix: tuple[TextureDraft, ...],
    final: TextureDraft,
    texture_budget: dict[str, Any],
) -> WholeScoreTextureResultV0 | None:
    try:
        result = assemble_whole_score_textures(
            "supported-normal-generation-v8-terminal-fallback",
            plan,
            skeleton,
            melodies,
            (*prefix, final),
            maximum_event_counts={
                item["material_id"]: item["maximum_texture_event_count"]
                for item in texture_budget["materials"]
            },
            placement_policy="bounded-backtracking-v4",
        )
    except ValueError:
        return None
    measurement = measure_score_texture_budget(plan, result.score, texture_budget)
    quality = evaluate_generic_score_quality(plan, result.score)
    if not measurement["whole_score"]["matches_budget"] or not quality["passes"]:
        return None
    return result


def repair_terminal_texture(
    plan: PiecePlan,
    skeleton: HarmonicSkeletonV0,
    melodies: ScorePayloadV0,
    prefix: tuple[TextureDraft, ...],
    source_final: TextureDraft,
    texture_budget: dict[str, Any],
) -> tuple[WholeScoreTextureResultV0, tuple[TextureDraft, ...], dict[str, Any]]:
    """v5最終伴奏を終止余白へ移し、固定予算に合う最初の解を返す。"""

    if len(prefix) != len(skeleton.materials) - 1:
        raise SupportedNormalGenerationTerminalFallbackError(
            "prepared texture prefix must contain all non-final materials"
        )
    final_budget = _final_material_budget(texture_budget)
    if (
        final_budget.get("length_units") != 67
        or final_budget.get("reserved_ending_units") != 7
        or final_budget.get("required_texture_event_count")
        != len(source_final.events)
    ):
        raise SupportedNormalGenerationTerminalFallbackError(
            "fixed final release budget does not match the source texture"
        )
    initial = _initial_terminal_events(source_final)
    final_melody = melodies.materials[-1]
    sizes = _combined_group_sizes(final_melody, initial)
    source_onsets = [onset for onset, size in sorted(sizes.items()) if size == 2]
    target_onsets = [onset for onset, size in sorted(sizes.items()) if size == 3]
    candidate_count = 0
    selected: dict[str, Any] | None = None
    selected_final: TextureDraft | None = None
    selected_result: WholeScoreTextureResultV0 | None = None
    for source_onset in source_onsets:
        source_indexes = [
            index
            for index, event in enumerate(initial)
            if event.at_units == source_onset
        ]
        for target_onset in target_onsets:
            if target_onset in {source_onset, TARGET_FINAL_ONSET_UNITS}:
                continue
            for source_index in source_indexes:
                event = initial[source_index]
                duration = min(
                    event.duration_units,
                    TARGET_FINAL_ONSET_UNITS - target_onset,
                )
                if duration <= 0:
                    continue
                candidate_count += 1
                candidate_events = tuple(
                    replace(
                        item,
                        at_units=target_onset,
                        duration_units=duration,
                    )
                    if index == source_index
                    else item
                    for index, item in enumerate(initial)
                )
                candidate_final = replace(source_final, events=candidate_events)
                result = _assemble_terminal_candidate(
                    plan,
                    skeleton,
                    melodies,
                    prefix,
                    candidate_final,
                    texture_budget,
                )
                if result is None:
                    continue
                selected = {
                    "source_onset_units": source_onset,
                    "target_onset_units": target_onset,
                    "source_event_index": source_index,
                    "degree": event.degree,
                }
                selected_final = candidate_final
                selected_result = result
                break
            if selected_result is not None:
                break
        if selected_result is not None:
            break
    if selected_result is None or selected_final is None or selected is None:
        raise SupportedNormalGenerationTerminalFallbackError(
            "deterministic terminal texture repair has no valid candidate"
        )
    invariants = {
        "event_count_unchanged": len(selected_final.events)
        == len(source_final.events),
        "degree_unchanged": Counter(event.degree for event in selected_final.events)
        == Counter(event.degree for event in source_final.events),
        "register_zone_unchanged": Counter(
            event.register_zone for event in selected_final.events
        )
        == Counter(event.register_zone for event in source_final.events),
    }
    if not all(invariants.values()):
        raise SupportedNormalGenerationTerminalFallbackError(
            "terminal texture repair changed protected content"
        )
    drafts = (*prefix, selected_final)
    return selected_result, drafts, {
        "schema_version": 1,
        "policy": "ending-slack-preserving-first-valid-candidate-v1",
        "source_final_onset_units": SOURCE_FINAL_ONSET_UNITS,
        "target_final_onset_units": TARGET_FINAL_ONSET_UNITS,
        "candidate_count": candidate_count,
        "selected": selected,
        "invariants": invariants,
    }


def _fixed_performance_drafts() -> tuple[PerformanceOccurrenceDraftV0, ...]:
    return (
        PerformanceOccurrenceDraftV0(
            "savor", "subtle", "shape", "light", "score", "harmony_legato"
        ),
        PerformanceOccurrenceDraftV0(
            "flow", "moderate", "shape", "light", "aligned", "harmony_legato"
        ),
        PerformanceOccurrenceDraftV0(
            "savor", "moderate", "shape", "legato", "aligned", "harmony_legato"
        ),
        PerformanceOccurrenceDraftV0(
            "build", "moderate", "build", "legato", "aligned", "harmony_legato"
        ),
        PerformanceOccurrenceDraftV0(
            "flow", "subtle", "shape", "light", "aligned", "harmony_legato"
        ),
        PerformanceOccurrenceDraftV0(
            "flow", "moderate", "release", "light", "score", "harmony_legato"
        ),
        PerformanceOccurrenceDraftV0(
            "release", "moderate", "release", "light", "aligned", "harmony_legato"
        ),
        PerformanceOccurrenceDraftV0(
            "release", "subtle", "release", "legato", "aligned", "harmony_legato"
        ),
    )


def build_terminal_performance(
    plan: PiecePlan,
    score: ScoreSpec,
    prompt_target: dict[str, Any],
):
    """固定profileを組み立て、既存のキー解放校正を再実行する。"""

    drafts = _fixed_performance_drafts()
    performance = assemble_whole_score_performance(
        "supported-normal-generation-v8-terminal-fallback",
        plan,
        score,
        drafts,
    )
    targets = [
        item
        for item in prompt_target.get("semantic_targets", {}).get(
            "rendered_surface", []
        )
        if item.get("id") == "key_held_texture"
    ]
    if len(targets) != 1:
        raise SupportedNormalGenerationTerminalFallbackError(
            "key-held texture target is missing"
        )
    performance, rendered, calibration = calibrate_key_release(
        plan,
        score,
        performance,
        targets[0],
    )
    return performance, rendered, calibration, {
        "schema_version": 1,
        "policy": "fixed-validated-performance-collection-v1",
        "source": "deterministic-preflight-on-fixed-score",
        "external_call_count": 0,
        "profiles": [asdict(item) for item in drafts],
    }


def _verify_v7_timeout_provenance(project_root: Path) -> dict[str, Any]:
    v7_root = project_root / V7_ROOT
    run_dir = v7_root / "runs/whole-score"
    outer = _read_json(v7_root / "manifest.json")
    inner = _read_json(run_dir / "manifest.json")
    state = _read_json(run_dir / "run-state.json")
    failure = _read_json(run_dir / "steps/run-failure.json")
    calls = state.get("calls", {})
    if (
        outer.get("status") != "failed"
        or outer.get("confirmed_external_call_count") != 1
        or inner.get("status") != "failed"
        or inner.get("confirmed_external_call_count") != 1
        or inner.get("error", {}).get("type") != "InterruptedAttemptError"
        or state.get("status") != "interrupted"
        or calls.get("confirmed_external_call_count") != 1
        or calls.get("interrupted_external_call_count") != 1
        or calls.get("successful_external_call_count") != 0
        or calls.get("saved_response_count") != 0
        or failure.get("status") != "failed"
        or failure.get("outputs", {}).get("type") != "InterruptedAttemptError"
        or (run_dir / "responses/texture-collection-batch-006.dsl").exists()
        or (run_dir / "outputs/performance-spec.dsl").exists()
        or (run_dir / "staged/final.mid").exists()
    ):
        raise SupportedNormalGenerationTerminalFallbackError(
            "v7 timeout provenance mismatch"
        )
    expected_hashes = {
        run_dir / "inputs/prepared-harmonic-collection.dsl": V6_HARMONY_SHA256,
        run_dir / "inputs/prepared-melody-collection.dsl": V6_MELODY_SHA256,
        run_dir / "inputs/prepared-texture-budget.json": V6_TEXTURE_BUDGET_SHA256,
        run_dir / "inputs/prepared-texture-prefix.dsl": V7_PREFIX_SHA256,
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise SupportedNormalGenerationTerminalFallbackError(
                f"v7 timeout provenance mismatch: {path.name}"
            )
    v5_run = project_root / V5_ROOT / "runs/whole-score"
    if (
        sha256_file(v5_run / "responses/texture-collection-batch-003.dsl")
        != V5_BATCH_3_RAW_SHA256
        or sha256_file(v5_run / "outputs/texture-collection-batch-003.dsl")
        != V5_BATCH_3_CANONICAL_SHA256
    ):
        raise SupportedNormalGenerationTerminalFallbackError(
            "v5 final texture provenance mismatch"
        )
    return {
        "schema_version": 1,
        "source_v7_status": "failed/interrupted",
        "confirmed_external_call_count": 1,
        "interrupted_external_call_count": 1,
        "successful_external_call_count": 0,
        "saved_response_count": 0,
    }


def prepare_terminal_fallback(
    project_root: Path,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """v7失敗を固定し、外部呼出し0回の新しいrunを準備する。"""

    project_root = Path(project_root).resolve()
    artifact_root = Path(artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = project_root / artifact_root
    artifact_root = artifact_root.resolve()
    provenance = _verify_v7_timeout_provenance(project_root)
    existing = artifact_root / "manifest.json"
    if existing.is_file():
        manifest = _read_json(existing)
        if manifest.get("protocol_id") != PROTOCOL_ID:
            raise SupportedNormalGenerationTerminalFallbackError(
                "saved terminal fallback manifest conflicts"
            )
        return manifest
    base = prepare_supported_normal_generation_repair(project_root, artifact_root)
    if base.get("status") != "prepared":
        raise SupportedNormalGenerationTerminalFallbackError(
            "base repair run was not prepared"
        )
    v5_batch = parse_texture_collection(
        (
            project_root
            / V5_ROOT
            / "runs/whole-score/outputs/texture-collection-batch-003.dsl"
        ).read_text(encoding="utf-8")
    )
    if len(v5_batch) != 3:
        raise SupportedNormalGenerationTerminalFallbackError(
            "v5 final texture collection count changed"
        )
    final_source = dump_texture_collection((v5_batch[-1],))
    final_source_path = artifact_root / "inputs/source-final-texture.dsl"
    _write_text(final_source_path, final_source)
    atomic_write_json(artifact_root / "inputs/v7-timeout-provenance.json", provenance)
    manifest = {
        **base,
        "protocol_id": PROTOCOL_ID,
        "status": "prepared",
        "passes": None,
        "promoted": False,
        "maximum_external_call_count": 0,
        "confirmed_external_call_count": 0,
        "source_final_texture_sha256": sha256_file(final_source_path),
        "v7_timeout_provenance_sha256": sha256_file(
            artifact_root / "inputs/v7-timeout-provenance.json"
        ),
        "fallback_implementation_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_json(existing, manifest)
    return manifest


def _save_texture_outputs(
    run_dir: Path,
    plan: PiecePlan,
    result: WholeScoreTextureResultV0,
    drafts: tuple[TextureDraft, ...],
    texture_budget: dict[str, Any],
    diagnostic: dict[str, Any],
) -> None:
    score_budget = measure_score_texture_budget(plan, result.score, texture_budget)
    score_quality = evaluate_generic_score_quality(plan, result.score)
    if not score_budget["whole_score"]["matches_budget"] or not score_quality["passes"]:
        raise SupportedNormalGenerationTerminalFallbackError(
            "terminal fallback score failed final validation"
        )
    texture_source = dump_texture_collection(drafts)
    final_source = dump_texture_collection((drafts[-1],))
    score_source = dump_score_spec(result.score)
    _write_text(run_dir / "outputs/texture-collection-batch-006.dsl", final_source)
    _write_text(run_dir / "outputs/texture-collection.dsl", texture_source)
    _write_text(run_dir / "outputs/score-spec.dsl", score_source)
    atomic_write_json(run_dir / "outputs/texture-budget.json", texture_budget)
    atomic_write_json(run_dir / "outputs/score-texture-budget.json", score_budget)
    atomic_write_json(run_dir / "outputs/generic-score-quality.json", score_quality)
    atomic_write_json(run_dir / "outputs/terminal-texture-repair.json", diagnostic)
    atomic_write_json(
        run_dir / "outputs/texture-placement.json",
        {
            "schema_version": 2,
            "placement_policy": "bounded-backtracking-v4",
            "placements": [asdict(item) for item in result.placements],
            "low_spacing_violations": list(result.low_spacing_violations),
        },
    )
    store = RunStore(run_dir, max_calls=0)
    store.record_step(
        "texture-terminal-fallback",
        "completed",
        {
            "source_final_texture": sha256_text(final_source),
            "texture_budget": sha256_json(texture_budget),
        },
        {
            "source_mode": "deterministic-fallback",
            "external_call_number": None,
            "diagnostic_sha256": sha256_json(diagnostic),
        },
    )
    store.record_step(
        "texture-collection",
        "completed",
        {
            "texture_budget": sha256_json(texture_budget),
            "terminal_repair": sha256_json(diagnostic),
        },
        {
            "texture_draft_sha256": sha256_text(texture_source),
            "score_spec_sha256": sha256_text(score_source),
            "score_texture_budget_sha256": sha256_json(score_budget),
            "material_count": len(drafts),
            "source_mode": "prepared-plus-deterministic-terminal-fallback",
            "external_call_number": None,
        },
    )


def _save_performance_outputs(
    run_dir: Path,
    plan: PiecePlan,
    score: ScoreSpec,
    prompt_target: dict[str, Any],
    texture_budget: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    performance, rendered, calibration, diagnostic = build_terminal_performance(
        plan,
        score,
        prompt_target,
    )
    rendered_budget = measure_rendered_texture_budget(rendered, texture_budget)
    score_quality = evaluate_generic_score_quality(plan, score)
    pipeline_quality = evaluate_generic_pipeline_quality(
        plan,
        score,
        performance,
        rendered,
    )
    if (
        not rendered_budget["rendered_performance"]["matches_budget"]
        or not score_quality["passes"]
        or not pipeline_quality["passes"]
    ):
        raise SupportedNormalGenerationTerminalFallbackError(
            "terminal fallback performance failed final validation"
        )
    drafts = _fixed_performance_drafts()
    performance_collection_source = dump_performance_collection(drafts)
    performance_source = dump_performance_spec(performance)
    _write_text(
        run_dir / "outputs/performance-collection.dsl",
        performance_collection_source,
    )
    _write_text(run_dir / "outputs/performance-spec.dsl", performance_source)
    atomic_write_json(run_dir / "outputs/key-release-calibration.json", calibration)
    atomic_write_json(run_dir / "outputs/fixed-performance-profile.json", diagnostic)
    atomic_write_json(
        run_dir / "outputs/rendered-texture-budget.json",
        rendered_budget,
    )
    atomic_write_json(
        run_dir / "outputs/generic-quality.json",
        {"score": score_quality, "pipeline": pipeline_quality},
    )
    staged_dir = run_dir / "staged"
    musicxml_path = render_musicxml(plan, score, staged_dir / "final.musicxml")
    smf_path = render_performance_smf(rendered, staged_dir / "final.mid").path
    RunStore(run_dir, max_calls=0).record_step(
        "performance-collection",
        "completed",
        {
            "score_spec": sha256_text(dump_score_spec(score)),
            "prompt_target": sha256_json(prompt_target),
            "texture_budget": sha256_json(texture_budget),
        },
        {
            "performance_spec_sha256": sha256_text(performance_source),
            "musicxml_sha256": sha256_file(musicxml_path),
            "smf_sha256": sha256_file(smf_path),
            "rendered_texture_budget_sha256": sha256_json(rendered_budget),
            "key_release_calibration_sha256": sha256_json(calibration),
            "source_mode": "deterministic-fallback",
            "external_call_number": None,
        },
    )
    return musicxml_path, smf_path, diagnostic


def run_terminal_fallback(
    project_root: Path,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """保存済み素材を再検証し、外部呼出しなしで最終評価まで通す。"""

    project_root = Path(project_root).resolve()
    artifact_root = Path(artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = project_root / artifact_root
    artifact_root = artifact_root.resolve()
    manifest = prepare_terminal_fallback(project_root, artifact_root)
    if manifest.get("status") != "prepared":
        return manifest
    run_dir = artifact_root / "runs/whole-score"
    runner = _NoExternalRunner()
    skeleton = execute_harmony_stage(project_root, run_dir, runner)
    melodies = execute_melody_stages(project_root, run_dir, runner, skeleton)
    plan = parse_piece_plan(
        (run_dir / "inputs/piece-plan.dsl").read_text(encoding="utf-8")
    )
    prefix = parse_texture_collection(
        (run_dir / "inputs/prepared-texture-prefix.dsl").read_text(
            encoding="utf-8"
        )
    )
    source_final_collection = parse_texture_collection(
        (artifact_root / "inputs/source-final-texture.dsl").read_text(
            encoding="utf-8"
        )
    )
    if len(source_final_collection) != 1:
        raise SupportedNormalGenerationTerminalFallbackError(
            "prepared final texture count changed"
        )
    texture_budget = _read_json(run_dir / "inputs/prepared-texture-budget.json")
    texture_result, drafts, texture_diagnostic = repair_terminal_texture(
        plan,
        skeleton,
        melodies,
        prefix,
        source_final_collection[0],
        texture_budget,
    )
    _save_texture_outputs(
        run_dir,
        plan,
        texture_result,
        drafts,
        texture_budget,
        texture_diagnostic,
    )
    prompt_target = _read_json(run_dir / "inputs/prompt-target.json")
    musicxml_path, smf_path, performance_diagnostic = _save_performance_outputs(
        run_dir,
        plan,
        texture_result.score,
        prompt_target,
        texture_budget,
    )
    evaluation = evaluate_staged_candidate(
        project_root,
        run_dir,
        candidate_use="feasibility_only",
    )
    manifest.update(
        {
            "status": evaluation["status"],
            "passes": bool(evaluation["passes"]),
            "promoted": bool(evaluation["promoted"]),
            "confirmed_external_call_count": 0,
            "texture_repair": texture_diagnostic,
            "performance_profile": performance_diagnostic,
            "staged_smf_sha256": sha256_file(smf_path),
            "staged_musicxml_sha256": sha256_file(musicxml_path),
            "evaluation": evaluation,
        }
    )
    final_smf = run_dir / "outputs/final.mid"
    final_musicxml = run_dir / "outputs/final.musicxml"
    if final_smf.is_file():
        manifest["smf_sha256"] = sha256_file(final_smf)
    if final_musicxml.is_file():
        manifest["musicxml_sha256"] = sha256_file(final_musicxml)
    atomic_write_json(artifact_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        result = prepare_terminal_fallback(
            arguments.project_root,
            arguments.artifact_root,
        )
        success = result.get("status") == "prepared"
    else:
        result = run_terminal_fallback(
            arguments.project_root,
            arguments.artifact_root,
        )
        success = (
            result.get("status") == "completed_unfit"
            and result.get("evaluation", {}).get("candidate_use")
            == "feasibility_only"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
