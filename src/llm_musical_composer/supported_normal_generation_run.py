"""広域候補から選んだ参照で、新規PiecePlanと3分SMFを一度生成する。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llm_musical_composer.generation_intent import merge_generation_intent
from llm_musical_composer.pilot_loop import (
    CodexExecRunner,
    isolated_codex_working_directory,
)
from llm_musical_composer.reference_generation_target import (
    build_reference_generation_target,
)
from llm_musical_composer.run_state import (
    RunStore,
    atomic_write_json,
    sha256_file,
    sha256_json,
)
from llm_musical_composer.staged_pipeline_generation import (
    StageRunner,
    run_staged_pipeline_generation,
    staged_pipeline_fingerprints,
)
from llm_musical_composer.supported_reference_selection import (
    select_supported_reference,
)
from llm_musical_composer.whole_score_staged_generation_run import (
    MODEL_CONFIG as WHOLE_SCORE_MODEL_CONFIG,
)
from llm_musical_composer.whole_score_staged_generation_run import (
    WholeScorePreparedSource,
    WholeScoreStageRunner,
    create_default_whole_score_runner,
    execute_prepared_whole_score_live_run,
    prepare_whole_score_live_run,
)

PROTOCOL_ID = "supported-normal-generation-v1"
DEFAULT_ARTIFACT_ROOT = Path(".appendix/supported-normal-generation-v1")
ANALYSIS_DIR = Path(".appendix/corpus-baseline-preflight-v1")
PIECE_PLAN_MODEL_CONFIG = {
    "model": "gpt-5.6-sol",
    "reasoning_effort": "high",
    "timeout_seconds": 900,
}
MAXIMUM_EXTERNAL_CALLS = 6
STABLE_STAGED_PROFILE_ID = "stable-staged-v1"
SEARCH_AWARE_STAGED_PROFILE_ID = "stable-staged-v2"
ATTACK_FREQUENCY_STAGED_PROFILE_ID = "stable-staged-v3"
STABLE_STAGED_PROFILE_IDS = frozenset(
    {
        STABLE_STAGED_PROFILE_ID,
        SEARCH_AWARE_STAGED_PROFILE_ID,
        ATTACK_FREQUENCY_STAGED_PROFILE_ID,
    }
)
STABLE_STAGED_SEED = "normal-generation-v2-seed-002"
STABLE_STAGED_SELECTION_SHA256 = (
    "0e7b3c95359ea801d51f6915b90b9a461860bd0cfee48e6ec638cc168beb3b9c"
)
STABLE_STAGED_TOTAL_EXTERNAL_CALLS = 10
STABLE_STAGED_WHOLE_SCORE_EXTERNAL_CALLS = 9
_IMPLEMENTATIONS = (
    Path("src/llm_musical_composer/supported_normal_generation_run.py"),
    Path("src/llm_musical_composer/supported_reference_selection.py"),
    Path("src/llm_musical_composer/reference_generation_target.py"),
    Path("src/llm_musical_composer/generation_intent.py"),
    Path("src/llm_musical_composer/tonal_hierarchy.py"),
    Path("src/llm_musical_composer/staged_pipeline_generation.py"),
    Path("src/llm_musical_composer/whole_score_staged_generation_run.py"),
    Path("src/llm_musical_composer/staged_material_pilot.py"),
    Path("src/llm_musical_composer/piano_texture_register_placement.py"),
    Path("src/llm_musical_composer/piano_texture_pilot.py"),
    Path("src/llm_musical_composer/pipeline_dsl.py"),
    Path("src/llm_musical_composer/performance_pipeline.py"),
)
_USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)


class SupportedNormalGenerationError(ValueError):
    """実験の不変入力または段階順序を再現できない。"""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupportedNormalGenerationError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise SupportedNormalGenerationError(f"JSON object required: {path}")
    return value


def _write_immutable_json(path: Path, value: Mapping[str, object], label: str) -> None:
    if path.exists():
        if _read_json(path) != value:
            raise SupportedNormalGenerationError(f"saved {label} conflicts with current input")
        return
    atomic_write_json(path, value)


def _implementation_hashes(project_root: Path) -> dict[str, str]:
    return {
        path.as_posix(): sha256_file(project_root / path) for path in _IMPLEMENTATIONS
    }


def _medoid_target(project_root: Path, resolved: Mapping[str, object]) -> dict[str, Any]:
    return build_reference_generation_target(
        resolved,
        reference_dir=project_root / ".appendix/reference-profile-v1",
        control_dir=project_root / ".appendix/control-reference-baseline-v3",
    ).artifact


def _profile_id(artifact_root: Path, requested: str | None) -> str:
    if requested is not None:
        profile_id = requested
    else:
        preflight_path = artifact_root / "preflight.json"
        if preflight_path.is_file():
            profile_id = str(
                _read_json(preflight_path).get("generation_profile_id", "legacy-v1")
            )
        else:
            profile_id = "legacy-v1"
    if profile_id not in {"legacy-v1", *STABLE_STAGED_PROFILE_IDS}:
        raise SupportedNormalGenerationError("generation profile is invalid")
    return profile_id


def _selection_parameters(
    project_root: Path,
    artifact_root: Path,
    profile_id: str,
    selection_seed: str | None,
    expected_selection_sha256: str | None,
) -> tuple[str | None, str | None]:
    if (selection_seed is None) != (expected_selection_sha256 is None):
        raise SupportedNormalGenerationError(
            "selection seed and expected selection SHA-256 must be given together"
        )
    if profile_id not in STABLE_STAGED_PROFILE_IDS:
        if selection_seed is not None:
            raise SupportedNormalGenerationError(
                "run-scoped selection is only valid for the stable staged profile"
            )
        return None, None

    preflight_path = artifact_root / "preflight.json"
    if preflight_path.is_file():
        preflight = _read_json(preflight_path)
        if preflight.get("schema_version") != 2:
            raise SupportedNormalGenerationError(
                "saved preflight schema cannot be resumed by this runner"
            )
        if preflight.get("generation_profile_id") != profile_id:
            raise SupportedNormalGenerationError(
                "saved generation profile conflicts with current input"
            )
        if preflight.get("implementation_hashes") != _implementation_hashes(
            project_root
        ):
            raise SupportedNormalGenerationError(
                "saved implementation hashes conflict with current input"
            )
        requested = preflight.get("requested_selection")
        if not isinstance(requested, Mapping):
            raise SupportedNormalGenerationError(
                "saved requested selection is invalid"
            )
        saved_seed = requested.get("seed")
        saved_sha256 = requested.get("expected_selection_sha256")
        if not isinstance(saved_seed, str) or not isinstance(saved_sha256, str):
            raise SupportedNormalGenerationError(
                "saved requested selection is invalid"
            )
        if selection_seed is not None and (
            selection_seed != saved_seed
            or expected_selection_sha256.casefold() != saved_sha256.casefold()
        ):
            raise SupportedNormalGenerationError(
                "saved requested selection conflicts with current input"
            )
        return saved_seed, saved_sha256

    if selection_seed is None:
        return STABLE_STAGED_SEED, STABLE_STAGED_SELECTION_SHA256
    return selection_seed, expected_selection_sha256


def _aggregate_usage(artifact_root: Path) -> dict[str, int]:
    result = {field: 0 for field in _USAGE_FIELDS}
    for relative in (
        Path("runs/piece-plan/run-state.json"),
        Path("runs/whole-score/run-state.json"),
    ):
        path = artifact_root / relative
        if not path.is_file():
            continue
        calls = _read_json(path).get("calls", {})
        if not isinstance(calls, Mapping):
            continue
        for field in _USAGE_FIELDS:
            try:
                result[field] += int(calls.get(field, 0))
            except (TypeError, ValueError) as error:
                raise SupportedNormalGenerationError(
                    "run token usage is invalid"
                ) from error
    return result


def _validate_total_external_calls(manifest: Mapping[str, object]) -> int:
    piece = int(manifest.get("piece_plan_external_call_count", 0))
    whole = int(manifest.get("whole_score_external_call_count", 0))
    total = piece + whole
    ceiling = int(manifest.get("maximum_external_call_count", MAXIMUM_EXTERNAL_CALLS))
    if total > ceiling:
        raise SupportedNormalGenerationError("total external call ceiling exceeded")
    return total


def prepare_supported_normal_generation(
    project_root: Path,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    *,
    profile_id: str | None = None,
    selection_seed: str | None = None,
    expected_selection_sha256: str | None = None,
    explicit_brightness: int | None = None,
    explicit_attack_frequency: float | None = None,
    generation_intent_path: Path | None = None,
) -> dict[str, Any]:
    """候補選択、匿名目標、対照、モデル上限を外部呼出し前に固定する。"""
    project_root = Path(project_root).resolve()
    artifact_root = Path(artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = project_root / artifact_root
    artifact_root = artifact_root.resolve()
    generation_intent = (
        _read_json(Path(generation_intent_path).resolve())
        if generation_intent_path is not None
        else None
    )
    generation_intent_sha256 = (
        sha256_json(generation_intent) if generation_intent is not None else None
    )
    preflight_path = artifact_root / "preflight.json"
    if preflight_path.is_file():
        saved_preflight = _read_json(preflight_path)
        saved_brightness = saved_preflight.get("internal_explicit_brightness")
        if saved_brightness != explicit_brightness:
            raise SupportedNormalGenerationError(
                "saved internal explicit brightness conflicts with current input"
            )
        if saved_preflight.get(
            "internal_explicit_attack_frequency"
        ) != explicit_attack_frequency:
            raise SupportedNormalGenerationError(
                "saved internal explicit attack frequency conflicts with current input"
            )
        if saved_preflight.get("generation_intent_sha256") != generation_intent_sha256:
            raise SupportedNormalGenerationError(
                "saved generation intent conflicts with current input"
            )
    profile_id = _profile_id(artifact_root, profile_id)
    selection_seed, expected_selection_sha256 = _selection_parameters(
        project_root,
        artifact_root,
        profile_id,
        selection_seed,
        expected_selection_sha256,
    )
    if profile_id in STABLE_STAGED_PROFILE_IDS:
        if selection_seed is None or expected_selection_sha256 is None:
            raise SupportedNormalGenerationError(
                "stable staged selection parameters are unavailable"
            )
        selected = select_supported_reference(
            project_root,
            project_root / ANALYSIS_DIR,
            seed=selection_seed,
            expected_selection_sha256=expected_selection_sha256,
            explicit_brightness=explicit_brightness,
            explicit_attack_frequency=explicit_attack_frequency,
        )
        maximum_external_calls = STABLE_STAGED_TOTAL_EXTERNAL_CALLS
    else:
        selected = select_supported_reference(
            project_root,
            project_root / ANALYSIS_DIR,
            explicit_brightness=explicit_brightness,
            explicit_attack_frequency=explicit_attack_frequency,
        )
        maximum_external_calls = MAXIMUM_EXTERNAL_CALLS
    medoid_target = _medoid_target(project_root, selected.medoid_control)
    inputs = artifact_root / "inputs"
    prompt_target = (
        merge_generation_intent(selected.target.prompt_target, generation_intent)
        if generation_intent is not None
        else selected.target.prompt_target
    )
    snapshots: dict[str, Mapping[str, object]] = {
        "selection.json": selected.selection,
        "normalized-request.json": selected.normalized_request,
        "resolved-request.json": selected.resolved_request,
        "target-artifact.json": selected.target.artifact,
        "prompt-target.json": prompt_target,
        "medoid-resolved-request.json": selected.medoid_control,
        "medoid-target-artifact.json": medoid_target,
    }
    if generation_intent is not None:
        snapshots["generation-intent.json"] = generation_intent
    for name, value in snapshots.items():
        _write_immutable_json(inputs / name, value, name)
    preflight = {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "status": "prepared",
        "internal_explicit_brightness": explicit_brightness,
        "internal_explicit_attack_frequency": explicit_attack_frequency,
        "generation_intent_sha256": generation_intent_sha256,
        "public_schema_connected": False,
        "generation_profile_id": profile_id,
        "requested_selection": (
            {
                "seed": selection_seed,
                "expected_selection_sha256": expected_selection_sha256,
            }
            if profile_id in STABLE_STAGED_PROFILE_IDS
            else None
        ),
        "selection": selected.selection,
        "medoid_control": selected.medoid_control,
        "piece_plan_model_config": PIECE_PLAN_MODEL_CONFIG,
        "whole_score_model_config": {
            **WHOLE_SCORE_MODEL_CONFIG,
            "maximum_external_calls": (
                STABLE_STAGED_WHOLE_SCORE_EXTERNAL_CALLS
                if profile_id in STABLE_STAGED_PROFILE_IDS
                else WHOLE_SCORE_MODEL_CONFIG["maximum_external_calls"]
            ),
        },
        "maximum_external_call_count": maximum_external_calls,
        "external_call_limits": (
            {"total": 10, "piece_plan": 1, "whole_score": 9}
            if profile_id in STABLE_STAGED_PROFILE_IDS
            else {"total": 6, "piece_plan": 1, "whole_score": 5}
        ),
        "texture_batch_policy": (
            {
                "maximum_event_count": 400,
                "maximum_batch_count": (
                    6 if profile_id == ATTACK_FREQUENCY_STAGED_PROFILE_ID else 4
                ),
            }
            if profile_id in STABLE_STAGED_PROFILE_IDS
            else {"maximum_event_count": None, "maximum_batch_count": None}
        ),
        "texture_placement_policy": (
            "search-aware-onset-zone-v8"
            if profile_id
            in {SEARCH_AWARE_STAGED_PROFILE_ID, ATTACK_FREQUENCY_STAGED_PROFILE_ID}
            else "onset-feasible-zone-v7"
            if profile_id == STABLE_STAGED_PROFILE_ID
            else "bounded-backtracking-v4"
        ),
        "velocity_policy_id": (
            "foreground-accompaniment-harmony-shape-v1"
            if profile_id in STABLE_STAGED_PROFILE_IDS
            else "legacy-unison-v1"
        ),
        "key_release_unreachable_policy": (
            "nearest_unfit" if profile_id in STABLE_STAGED_PROFILE_IDS else "raise"
        ),
        "retry_policy": "none",
        "implementation_hashes": _implementation_hashes(project_root),
        "staged_pipeline_fingerprints": staged_pipeline_fingerprints(),
        "input_hashes": {
            name: sha256_file(inputs / name) for name in sorted(snapshots)
        },
    }
    _write_immutable_json(artifact_root / "preflight.json", preflight, "preflight")
    manifest_path = artifact_root / "manifest.json"
    if manifest_path.exists():
        return _read_json(manifest_path)
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "generation_profile_id": profile_id,
        "status": "prepared",
        "passes": None,
        "promoted": False,
        "maximum_external_call_count": maximum_external_calls,
        "preflight_sha256": sha256_file(artifact_root / "preflight.json"),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _default_piece_plan_runner(run_dir: Path) -> StageRunner:
    return CodexExecRunner(
        run_store=RunStore(run_dir, max_calls=1),
        schema_path=Path(__file__).resolve().parents[2]
        / "schemas"
        / "codex-composition-response.schema.json",
        model=PIECE_PLAN_MODEL_CONFIG["model"],
        reasoning_effort=PIECE_PLAN_MODEL_CONFIG["reasoning_effort"],
        working_directory=isolated_codex_working_directory(),
        timeout_seconds=PIECE_PLAN_MODEL_CONFIG["timeout_seconds"],
    )


def execute_supported_piece_plan(
    project_root: Path,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    *,
    runner: StageRunner | None = None,
    profile_id: str | None = None,
    selection_seed: str | None = None,
    expected_selection_sha256: str | None = None,
    explicit_brightness: int | None = None,
    explicit_attack_frequency: float | None = None,
    generation_intent_path: Path | None = None,
) -> dict[str, Any]:
    """新規PiecePlanを一回生成し、合格時だけ曲全体runを準備する。"""
    project_root = Path(project_root).resolve()
    artifact_root = Path(artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = project_root / artifact_root
    artifact_root = artifact_root.resolve()
    manifest = prepare_supported_normal_generation(
        project_root,
        artifact_root,
        profile_id=profile_id,
        selection_seed=selection_seed,
        expected_selection_sha256=expected_selection_sha256,
        explicit_brightness=explicit_brightness,
        explicit_attack_frequency=explicit_attack_frequency,
        generation_intent_path=generation_intent_path,
    )
    if manifest.get("status") != "prepared":
        return manifest
    inputs = artifact_root / "inputs"
    piece_run = artifact_root / "runs" / "piece-plan"
    active_runner = runner or _default_piece_plan_runner(piece_run)
    try:
        result = run_staged_pipeline_generation(
            run_dir=piece_run,
            normalized_request=_read_json(inputs / "normalized-request.json"),
            resolved_request=_read_json(inputs / "resolved-request.json"),
            prompt_target=_read_json(inputs / "prompt-target.json"),
            model_config=PIECE_PLAN_MODEL_CONFIG,
            runner=active_runner,
            stop_after="piece_plan",
        )
        if result.status != "stopped" or result.stopped_after != "piece_plan":
            raise SupportedNormalGenerationError("PiecePlan stage did not stop as prepared")
        generation_intent_snapshot = inputs / "generation-intent.json"
        if generation_intent_snapshot.is_file():
            _write_immutable_json(
                piece_run / "inputs/generation-intent.json",
                _read_json(generation_intent_snapshot),
                "PiecePlan generation intent",
            )
        whole_run = artifact_root / "runs" / "whole-score"
        selected_profile = str(manifest.get("generation_profile_id", "legacy-v1"))
        source = (
            WholeScorePreparedSource(
                run_root=piece_run,
                texture_batch_maximum_event_count=400,
                texture_batch_maximum_count=(
                    6
                    if selected_profile == ATTACK_FREQUENCY_STAGED_PROFILE_ID
                    else 4
                ),
                maximum_external_calls=STABLE_STAGED_WHOLE_SCORE_EXTERNAL_CALLS,
                generation_profile_id=selected_profile,
                velocity_policy_id=(
                    "foreground-accompaniment-harmony-shape-v1"
                ),
                key_release_unreachable_policy="nearest_unfit",
            )
            if selected_profile in STABLE_STAGED_PROFILE_IDS
            else WholeScorePreparedSource(run_root=piece_run)
        )
        prepared = prepare_whole_score_live_run(
            project_root,
            whole_run,
            source=source,
        )
        manifest.update(
            {
                "status": "whole_score_prepared",
                "piece_plan_external_call_count": active_runner.call_number,
                "piece_plan_path": str(result.piece_plan_path.relative_to(artifact_root)),
                "piece_plan_sha256": sha256_file(result.piece_plan_path),
                "whole_score_preflight": prepared,
            }
        )
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "passes": False,
                "promoted": False,
                "piece_plan_external_call_count": active_runner.call_number,
                "error": {"type": type(error).__name__, "detail": str(error)},
            }
        )
    manifest["usage"] = _aggregate_usage(artifact_root)
    atomic_write_json(artifact_root / "manifest.json", manifest)
    return manifest


def execute_supported_whole_score(
    project_root: Path,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    *,
    runner: WholeScoreStageRunner | None = None,
) -> dict[str, Any]:
    """準備済みPiecePlanを曲全体生成へ渡し、結果を外側manifestへ確定する。"""
    project_root = Path(project_root).resolve()
    artifact_root = Path(artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = project_root / artifact_root
    artifact_root = artifact_root.resolve()
    manifest = _read_json(artifact_root / "manifest.json")
    if manifest.get("status") != "whole_score_prepared":
        return manifest
    whole_run = artifact_root / "runs" / "whole-score"
    active_runner = runner or create_default_whole_score_runner(project_root, whole_run)
    result = execute_prepared_whole_score_live_run(
        project_root,
        whole_run,
        active_runner,
    )
    manifest.update(
        {
            "status": result["status"],
            "passes": bool(result.get("passes")),
            "promoted": bool(result.get("promoted")),
            "whole_score_external_call_count": active_runner.call_number,
            "confirmed_external_call_count": (
                int(manifest.get("piece_plan_external_call_count", 0))
                + active_runner.call_number
            ),
            "whole_score_result": result,
        }
    )
    manifest["confirmed_external_call_count"] = _validate_total_external_calls(
        manifest
    )
    manifest["usage"] = _aggregate_usage(artifact_root)
    final_midi = whole_run / "outputs/final.mid"
    if final_midi.is_file():
        manifest["final_smf_path"] = str(final_midi.relative_to(artifact_root))
        manifest["final_smf_sha256"] = sha256_file(final_midi)
    atomic_write_json(artifact_root / "manifest.json", manifest)
    return manifest


def run_supported_normal_generation(
    project_root: Path,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    *,
    profile_id: str | None = None,
    selection_seed: str | None = None,
    expected_selection_sha256: str | None = None,
    explicit_brightness: int | None = None,
    explicit_attack_frequency: float | None = None,
    generation_intent_path: Path | None = None,
) -> dict[str, Any]:
    """準備済みでない段階だけを順に進め、自動再試行せず終端する。"""
    manifest = prepare_supported_normal_generation(
        project_root,
        artifact_root,
        profile_id=profile_id,
        selection_seed=selection_seed,
        expected_selection_sha256=expected_selection_sha256,
        explicit_brightness=explicit_brightness,
        explicit_attack_frequency=explicit_attack_frequency,
        generation_intent_path=generation_intent_path,
    )
    if manifest.get("status") == "prepared":
        manifest = execute_supported_piece_plan(
            project_root,
            artifact_root,
            profile_id=profile_id,
            selection_seed=selection_seed,
            expected_selection_sha256=expected_selection_sha256,
            explicit_brightness=explicit_brightness,
            explicit_attack_frequency=explicit_attack_frequency,
            generation_intent_path=generation_intent_path,
        )
    if manifest.get("status") == "whole_score_prepared":
        manifest = execute_supported_whole_score(project_root, artifact_root)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--profile-id",
        choices=("legacy-v1", *sorted(STABLE_STAGED_PROFILE_IDS)),
        default=None,
    )
    parser.add_argument("--selection-seed", default=None)
    parser.add_argument("--expected-selection-sha256", default=None)
    parser.add_argument(
        "--internal-explicit-brightness",
        type=int,
        choices=(-1, 0, 1),
        default=None,
        help="内部診断専用。公開Schemaには接続しない。",
    )
    parser.add_argument(
        "--generation-intent",
        type=Path,
        default=None,
        help="内部生成方針JSON。公開Schemaには接続しない。",
    )
    parser.add_argument(
        "--internal-explicit-attack-frequency",
        type=float,
        default=None,
        help="内部診断専用。公開Schemaには接続しない。",
    )
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        result = prepare_supported_normal_generation(
            arguments.project_root,
            arguments.artifact_root,
            profile_id=arguments.profile_id,
            selection_seed=arguments.selection_seed,
            expected_selection_sha256=arguments.expected_selection_sha256,
            explicit_brightness=arguments.internal_explicit_brightness,
            explicit_attack_frequency=arguments.internal_explicit_attack_frequency,
            generation_intent_path=arguments.generation_intent,
        )
        success = result.get("status") == "prepared"
    else:
        result = run_supported_normal_generation(
            arguments.project_root,
            arguments.artifact_root,
            profile_id=arguments.profile_id,
            selection_seed=arguments.selection_seed,
            expected_selection_sha256=arguments.expected_selection_sha256,
            explicit_brightness=arguments.internal_explicit_brightness,
            explicit_attack_frequency=arguments.internal_explicit_attack_frequency,
            generation_intent_path=arguments.generation_intent,
        )
        success = result.get("status") == "completed_fit"
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
