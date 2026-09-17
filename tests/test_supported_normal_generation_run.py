import json
from pathlib import Path

import pytest

from llm_musical_composer.supported_normal_generation_run import (
    ATTACK_FREQUENCY_STAGED_PROFILE_ID,
    SEARCH_AWARE_STAGED_PROFILE_ID,
    STABLE_STAGED_PROFILE_ID,
    SupportedNormalGenerationError,
    execute_supported_piece_plan,
    prepare_supported_normal_generation,
)
from llm_musical_composer.whole_score_staged_generation_run import (
    WholeScoreLiveRunError,
    WholeScorePreparedSource,
    prepare_whole_score_live_run,
)

PROJECT_ROOT = Path(__file__).parents[1]
PLAN_PATH = (
    PROJECT_ROOT
    / ".appendix/reference-variance-smoke-v7/runs/reference-a-candidate-2/outputs/piece-plan.dsl"
)
SEED_003 = "normal-generation-v2-seed-003"
SELECTION_SHA256_003 = (
    "279bfadd11c746a78bf01fa70ed21f1191da7b5fcb7d05b0138a325c9e9f1894"
)


class _PiecePlanRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def call_number(self) -> int:
        return len(self.calls)

    def run(self, step_id: str, prompt: str, input_hashes=None):
        del prompt, input_hashes
        self.calls.append(step_id)
        return {
            "composition_source": PLAN_PATH.read_text(encoding="utf-8"),
            "intent_summary": "fixture",
        }


def _generation_intent() -> dict[str, object]:
    return {
        "schema_version": 1,
        "decisions": [
            {
                "decision_id": "harmonic_contrast",
                "statement": "区分ごとの和声進行を区別する",
                "decision_kind": "section_harmonic_contrast",
                "stage_targets": [
                    {
                        "generation_stage": "harmonic_skeleton",
                        "instructions": ["区分ごとに異なる和声機能を使う"],
                        "observable": {
                            "observable_id": "section_harmonic_progression",
                            "scope": "whole_piece",
                            "operator": "differentiate",
                        },
                        "verification": "mechanical",
                    }
                ],
            }
        ],
    }


def test_prepare_fixes_selection_targets_and_call_ceiling(tmp_path: Path) -> None:
    artifact_root = tmp_path / "supported"

    result = prepare_supported_normal_generation(PROJECT_ROOT, artifact_root)

    assert result["status"] == "prepared"
    preflight = json.loads(
        (artifact_root / "preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["maximum_external_call_count"] == 6
    assert preflight["selection"]["selected_name"] == "mayodance.mid"
    assert preflight["selection"]["candidate_count"] == 171
    assert preflight["medoid_control"]["reference"]["name"] == (
        "SonataForHouseMoving.mid"
    )
    prompt_target = json.loads(
        (artifact_root / "inputs/prompt-target.json").read_text(encoding="utf-8")
    )
    frequency = next(
        item
        for item in prompt_target["semantic_targets"]["score_spec"]
        if item["id"] == "attack_frequency"
    )
    assert frequency["strict_score_budget"]["groups"] == 649
    assert frequency["groups_for_180_seconds"] == {
        "anchor_half_up": 816,
        "minimum_ceil": 649,
        "maximum_floor": 1058,
    }


def test_prepare_snapshots_and_merges_generation_intent(tmp_path: Path) -> None:
    artifact_root = tmp_path / "intent"
    intent_path = tmp_path / "generation-intent.json"
    intent = _generation_intent()
    intent_path.write_text(
        json.dumps(intent, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    prepare_supported_normal_generation(
        PROJECT_ROOT,
        artifact_root,
        generation_intent_path=intent_path,
    )

    assert json.loads(
        (artifact_root / "inputs/generation-intent.json").read_text(encoding="utf-8")
    ) == intent
    target = json.loads(
        (artifact_root / "inputs/prompt-target.json").read_text(encoding="utf-8")
    )
    creative = [
        item
        for item in target["semantic_targets"]["score_spec"]
        if item.get("kind") == "creative_intent"
    ]
    assert [item["generation_stage"] for item in creative] == ["harmonic_skeleton"]
    preflight = json.loads((artifact_root / "preflight.json").read_text(encoding="utf-8"))
    assert "generation-intent.json" in preflight["input_hashes"]
    assert (
        "src/llm_musical_composer/generation_intent.py"
        in preflight["implementation_hashes"]
    )

    with pytest.raises(SupportedNormalGenerationError, match="generation intent conflicts"):
        prepare_supported_normal_generation(PROJECT_ROOT, artifact_root)


def test_generation_intent_survives_piece_plan_to_whole_score_prepare(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "intent-whole-score"
    intent_path = tmp_path / "generation-intent.json"
    intent = _generation_intent()
    intent_path.write_text(
        json.dumps(intent, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = execute_supported_piece_plan(
        PROJECT_ROOT,
        artifact_root,
        runner=_PiecePlanRunner(),
        generation_intent_path=intent_path,
    )

    assert result["status"] == "whole_score_prepared"
    piece_run = artifact_root / "runs/piece-plan"
    whole_run = artifact_root / "runs/whole-score"
    assert json.loads(
        (piece_run / "inputs/generation-intent.json").read_text(encoding="utf-8")
    ) == intent
    target = json.loads(
        (whole_run / "inputs/prompt-target.json").read_text(encoding="utf-8")
    )
    assert target["generation_intent"]["decision_ids"] == ["harmonic_contrast"]
    assert json.loads(
        (whole_run / "inputs/generation-intent.json").read_text(encoding="utf-8")
    ) == intent
    spec = json.loads((whole_run / "run-spec.json").read_text(encoding="utf-8"))
    assert "generation_intent" in spec["input_hashes"]["generation_inputs"]

    intent["decisions"][0]["statement"] = "差し替えた方針"
    (piece_run / "inputs/generation-intent.json").write_text(
        json.dumps(intent, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(WholeScoreLiveRunError, match="generation intent hash mismatch"):
        prepare_whole_score_live_run(
            PROJECT_ROOT,
            tmp_path / "tampered-whole-score",
            source=WholeScorePreparedSource(run_root=piece_run),
        )


def test_prepare_internal_brightness_zero_is_idempotent_and_provenanced(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "brightness-zero"

    first = prepare_supported_normal_generation(
        PROJECT_ROOT,
        artifact_root,
        explicit_brightness=0,
    )
    second = prepare_supported_normal_generation(
        PROJECT_ROOT,
        artifact_root,
        explicit_brightness=0,
    )

    assert first == second
    preflight = json.loads((artifact_root / "preflight.json").read_text(encoding="utf-8"))
    assert preflight["internal_explicit_brightness"] == 0
    assert preflight["public_schema_connected"] is False
    normalized = json.loads(
        (artifact_root / "inputs/normalized-request.json").read_text(encoding="utf-8")
    )
    resolved = json.loads(
        (artifact_root / "inputs/resolved-request.json").read_text(encoding="utf-8")
    )
    target = json.loads(
        (artifact_root / "inputs/prompt-target.json").read_text(encoding="utf-8")
    )
    assert normalized["controls"]["brightness"]["value"] == 0.0
    assert resolved["controls"] == normalized["controls"]
    assert target["semantic_targets"]["piece_plan"][0]["scale_policy"] == "dorian"

    with pytest.raises(SupportedNormalGenerationError, match="brightness conflicts"):
        prepare_supported_normal_generation(PROJECT_ROOT, artifact_root)


def test_prepare_internal_attack_frequency_zero_is_idempotent_and_uses_six_batches(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "frequency-zero"

    first = prepare_supported_normal_generation(
        PROJECT_ROOT,
        artifact_root,
        profile_id=ATTACK_FREQUENCY_STAGED_PROFILE_ID,
        explicit_attack_frequency=0.0,
    )
    second = prepare_supported_normal_generation(
        PROJECT_ROOT,
        artifact_root,
        profile_id=ATTACK_FREQUENCY_STAGED_PROFILE_ID,
        explicit_attack_frequency=0.0,
    )

    assert first == second
    preflight = json.loads((artifact_root / "preflight.json").read_text(encoding="utf-8"))
    assert preflight["internal_explicit_attack_frequency"] == 0.0
    assert preflight["texture_batch_policy"] == {
        "maximum_event_count": 400,
        "maximum_batch_count": 6,
    }
    target = json.loads(
        (artifact_root / "inputs/prompt-target.json").read_text(encoding="utf-8")
    )
    frequency = next(
        item
        for item in target["semantic_targets"]["score_spec"]
        if item["id"] == "attack_frequency"
    )
    assert frequency["strict_score_budget"]["groups"] == 752

    with pytest.raises(SupportedNormalGenerationError, match="frequency conflicts"):
        prepare_supported_normal_generation(
            PROJECT_ROOT,
            artifact_root,
            profile_id=ATTACK_FREQUENCY_STAGED_PROFILE_ID,
        )


def test_piece_plan_execution_propagates_internal_attack_frequency_to_whole_score(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "frequency-zero"
    prepare_supported_normal_generation(
        PROJECT_ROOT,
        artifact_root,
        profile_id=ATTACK_FREQUENCY_STAGED_PROFILE_ID,
        explicit_attack_frequency=0.0,
    )

    result = execute_supported_piece_plan(
        PROJECT_ROOT,
        artifact_root,
        runner=_PiecePlanRunner(),
        profile_id=ATTACK_FREQUENCY_STAGED_PROFILE_ID,
        explicit_attack_frequency=0.0,
    )

    assert result["status"] == "whole_score_prepared"
    spec = json.loads(
        (artifact_root / "runs/whole-score/run-spec.json").read_text(
            encoding="utf-8"
        )
    )
    assert spec["generation_profile_id"] == ATTACK_FREQUENCY_STAGED_PROFILE_ID
    assert spec["texture_placement_policy"] == "search-aware-onset-zone-v8"
    assert spec["texture_batch_maximum_count"] == 6
    assert spec["model_config"]["maximum_external_calls"] == 9
    assert spec["attack_frequency"] == {
        "target": 4.17764351,
        "candidate_minimum": 4.175,
        "candidate_maximum": 4.18055556,
        "minimum_attack_group_count": 752,
        "group_tolerance_ms": 30,
        "source": "explicit_corpus_min_max_v1",
    }


def test_piece_plan_execution_propagates_internal_brightness_to_second_prepare(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "brightness-zero"
    prepare_supported_normal_generation(
        PROJECT_ROOT,
        artifact_root,
        explicit_brightness=0,
    )
    runner = _PiecePlanRunner()

    result = execute_supported_piece_plan(
        PROJECT_ROOT,
        artifact_root,
        runner=runner,
        explicit_brightness=0,
    )

    assert result["status"] == "failed"
    assert "tonal-hierarchy:plan-mode" in result["error"]["detail"]
    assert runner.calls == ["piece-plan"]


def test_piece_plan_stage_prepares_dynamic_whole_score_input(tmp_path: Path) -> None:
    artifact_root = tmp_path / "supported"
    prepare_supported_normal_generation(PROJECT_ROOT, artifact_root)
    runner = _PiecePlanRunner()

    result = execute_supported_piece_plan(
        PROJECT_ROOT,
        artifact_root,
        runner=runner,
    )

    assert result["status"] == "whole_score_prepared"
    assert result["usage"] == {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    assert runner.calls == ["piece-plan"]
    whole_run = artifact_root / "runs" / "whole-score"
    spec = json.loads((whole_run / "run-spec.json").read_text(encoding="utf-8"))
    assert spec["attack_frequency"] == {
        "target": 4.5357329,
        "candidate_minimum": 3.60069793,
        "candidate_maximum": 5.87968569,
        "minimum_attack_group_count": 649,
        "group_tolerance_ms": 30,
        "source": "reference_neighborhood_v3",
    }
    assert spec["input_hashes"]["evaluation_inputs"].keys() == {
        "reference_profiles",
        "capability_v26_smf",
        "multiscale_v8_smf",
    }
    assert json.loads((whole_run / "manifest.json").read_text(encoding="utf-8"))[
        "source_run_status"
    ] == "running"


def test_stable_profile_fixes_new_seed_and_full_pipeline_limits(tmp_path: Path) -> None:
    artifact_root = tmp_path / "stable"

    result = prepare_supported_normal_generation(
        PROJECT_ROOT,
        artifact_root,
        profile_id=STABLE_STAGED_PROFILE_ID,
    )

    assert result["status"] == "prepared"
    preflight = json.loads(
        (artifact_root / "preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["generation_profile_id"] == STABLE_STAGED_PROFILE_ID
    assert preflight["selection"]["seed"] == "normal-generation-v2-seed-002"
    assert preflight["selection"]["selection_sha256"] == (
        "0e7b3c95359ea801d51f6915b90b9a461860bd0cfee48e6ec638cc168beb3b9c"
    )
    assert preflight["selection"]["selected_name"] == "oldkey.mid"
    assert preflight["external_call_limits"] == {
        "total": 10,
        "piece_plan": 1,
        "whole_score": 9,
    }
    assert preflight["whole_score_model_config"]["maximum_external_calls"] == 9
    assert preflight["texture_batch_policy"] == {
        "maximum_event_count": 400,
        "maximum_batch_count": 4,
    }
    assert preflight["velocity_policy_id"] == (
        "foreground-accompaniment-harmony-shape-v1"
    )
    assert preflight["key_release_unreachable_policy"] == "nearest_unfit"


def test_stable_profile_accepts_run_scoped_selection(tmp_path: Path) -> None:
    artifact_root = tmp_path / "stable-seed-003"

    result = prepare_supported_normal_generation(
        PROJECT_ROOT,
        artifact_root,
        profile_id=STABLE_STAGED_PROFILE_ID,
        selection_seed=SEED_003,
        expected_selection_sha256=SELECTION_SHA256_003,
    )

    assert result["status"] == "prepared"
    preflight = json.loads(
        (artifact_root / "preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["schema_version"] == 2
    assert preflight["requested_selection"] == {
        "seed": SEED_003,
        "expected_selection_sha256": SELECTION_SHA256_003,
    }
    assert preflight["selection"]["selected_name"] == "R047_003.MID"
    assert preflight["selection"]["selected_reference_sha256"] == (
        "af862b0b2701c0359cb8e01fe07fcfd7db465cc6144d862c02a021d825d4e7a4"
    )


@pytest.mark.parametrize(
    ("selection_seed", "expected_selection_sha256"),
    ((SEED_003, None), (None, SELECTION_SHA256_003)),
)
def test_selection_arguments_must_be_given_together(
    tmp_path: Path,
    selection_seed: str | None,
    expected_selection_sha256: str | None,
) -> None:
    with pytest.raises(SupportedNormalGenerationError, match="given together"):
        prepare_supported_normal_generation(
            PROJECT_ROOT,
            tmp_path / "invalid",
            profile_id=STABLE_STAGED_PROFILE_ID,
            selection_seed=selection_seed,
            expected_selection_sha256=expected_selection_sha256,
        )


def test_legacy_profile_rejects_selection_arguments(tmp_path: Path) -> None:
    with pytest.raises(SupportedNormalGenerationError, match="stable staged"):
        prepare_supported_normal_generation(
            PROJECT_ROOT,
            tmp_path / "legacy",
            profile_id="legacy-v1",
            selection_seed=SEED_003,
            expected_selection_sha256=SELECTION_SHA256_003,
        )


def test_saved_selection_is_reused_and_conflicts_before_snapshot_write(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "stable-seed-003"
    prepare_supported_normal_generation(
        PROJECT_ROOT,
        artifact_root,
        profile_id=STABLE_STAGED_PROFILE_ID,
        selection_seed=SEED_003,
        expected_selection_sha256=SELECTION_SHA256_003,
    )
    selection_path = artifact_root / "inputs/selection.json"
    saved_selection = selection_path.read_bytes()

    result = execute_supported_piece_plan(
        PROJECT_ROOT,
        artifact_root,
        runner=_PiecePlanRunner(),
    )

    assert result["status"] == "whole_score_prepared"
    spec = json.loads(
        (artifact_root / "runs/whole-score/run-spec.json").read_text(
            encoding="utf-8"
        )
    )
    assert spec["texture_placement_policy"] == "onset-feasible-zone-v7"
    required_hashes = {
        "src/llm_musical_composer/piano_texture_register_placement.py",
        "src/llm_musical_composer/staged_material_pilot.py",
        "src/llm_musical_composer/piano_texture_pilot.py",
    }
    preflight = json.loads(
        (artifact_root / "preflight.json").read_text(encoding="utf-8")
    )
    assert required_hashes <= preflight["implementation_hashes"].keys()
    assert required_hashes <= spec["implementation_hashes"].keys()

    with pytest.raises(SupportedNormalGenerationError, match="conflicts"):
        prepare_supported_normal_generation(
            PROJECT_ROOT,
            artifact_root,
            profile_id=STABLE_STAGED_PROFILE_ID,
            selection_seed="normal-generation-v2-seed-002",
            expected_selection_sha256=(
                "0e7b3c95359ea801d51f6915b90b9a461860bd0cfee48e6ec638cc168beb3b9c"
            ),
        )
    assert selection_path.read_bytes() == saved_selection


def test_stable_profile_propagates_to_whole_score_run(tmp_path: Path) -> None:
    artifact_root = tmp_path / "stable"
    prepare_supported_normal_generation(
        PROJECT_ROOT,
        artifact_root,
        profile_id=STABLE_STAGED_PROFILE_ID,
    )

    result = execute_supported_piece_plan(
        PROJECT_ROOT,
        artifact_root,
        runner=_PiecePlanRunner(),
    )

    assert result["status"] == "whole_score_prepared"
    spec = json.loads(
        (artifact_root / "runs/whole-score/run-spec.json").read_text(
            encoding="utf-8"
        )
    )
    assert spec["generation_profile_id"] == STABLE_STAGED_PROFILE_ID
    assert spec["model_config"]["maximum_external_calls"] == 9
    assert spec["texture_batch_maximum_event_count"] == 400
    assert spec["texture_batch_maximum_count"] == 4
    assert spec["velocity_policy_id"] == (
        "foreground-accompaniment-harmony-shape-v1"
    )
    assert spec["key_release_unreachable_policy"] == "nearest_unfit"


def test_search_aware_profile_fixes_v8_without_changing_v1(tmp_path: Path) -> None:
    artifact_root = tmp_path / "stable-v2"

    prepare_supported_normal_generation(
        PROJECT_ROOT,
        artifact_root,
        profile_id=SEARCH_AWARE_STAGED_PROFILE_ID,
        selection_seed=SEED_003,
        expected_selection_sha256=SELECTION_SHA256_003,
    )
    preflight = json.loads(
        (artifact_root / "preflight.json").read_text(encoding="utf-8")
    )
    result = execute_supported_piece_plan(
        PROJECT_ROOT,
        artifact_root,
        runner=_PiecePlanRunner(),
    )

    assert preflight["generation_profile_id"] == SEARCH_AWARE_STAGED_PROFILE_ID
    assert preflight["texture_placement_policy"] == "search-aware-onset-zone-v8"
    assert result["status"] == "whole_score_prepared"
    spec = json.loads(
        (artifact_root / "runs/whole-score/run-spec.json").read_text(
            encoding="utf-8"
        )
    )
    assert spec["generation_profile_id"] == SEARCH_AWARE_STAGED_PROFILE_ID
    assert spec["texture_placement_policy"] == "search-aware-onset-zone-v8"
