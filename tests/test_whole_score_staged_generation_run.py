from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import llm_musical_composer.whole_score_staged_generation_run as staged_run
from llm_musical_composer.generation_intent import (
    creative_targets_sha256,
    merge_generation_intent,
)
from llm_musical_composer.harmonic_skeleton import apply_harmonic_skeleton
from llm_musical_composer.performance_pipeline import (
    HARMONY_INTERVALS,
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PlanNode,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)
from llm_musical_composer.piano_texture_register_placement import (
    PianoTexturePlacementResultV5,
    PianoTexturePlacementResultV6,
    PianoTexturePlacementResultV7,
    PianoTexturePlacementResultV8,
)
from llm_musical_composer.pipeline_dsl import (
    dump_piece_plan,
    dump_score_spec,
    parse_piece_plan,
)
from llm_musical_composer.reference_generation_target import (
    build_reference_generation_target,
)
from llm_musical_composer.run_state import RunStore
from llm_musical_composer.staged_material_pilot import (
    HarmonicDraft,
    HarmonicEventDraft,
    MelodyDraft,
    MelodyEventDraft,
    TextureDraft,
    TextureEventDraft,
    assemble_texture,
    full_low_spacing_violations,
)
from llm_musical_composer.texture_budget import allocate_texture_budget
from llm_musical_composer.whole_score_staged_generation import (
    PerformanceOccurrenceDraftV0,
    WholeHarmonicMaterialDraftV0,
    assemble_whole_score_melodies,
    assemble_whole_score_skeleton,
)
from llm_musical_composer.whole_score_staged_generation_dsl import (
    dump_harmonic_collection,
    dump_melody_collection,
    dump_performance_collection,
    dump_texture_collection,
    parse_harmonic_collection,
    parse_melody_collection,
    parse_texture_collection,
)
from llm_musical_composer.whole_score_staged_generation_run import (
    REGISTER_LOWER_BOUND_EXPERIMENT_POLICY_ID,
    WholeScoreLiveRunError,
    WholeScorePreparedSource,
    _harmonic_stage_context,
    _melody_semantic_targets,
    _prompt_target_for_saved_version,
    _rendered_texture_budget_passes,
    _save_stage_failure,
    _semantic_texture_fit,
    _TextureBatchPlacementError,
    apply_register_promotion_policy,
    build_allowed_pitch_range,
    build_register_exception_counterfactual,
    build_register_exception_provenance,
    calibrate_default_velocity,
    calibrate_key_release,
    evaluate_staged_candidate,
    execute_harmony_stage,
    execute_melody_stages,
    execute_performance_stage,
    execute_prepared_whole_score_live_run,
    execute_texture_stage,
    maximum_attack_group_capacity,
    measure_register_diagnostic,
    measure_register_progress,
    normalize_harmonic_capacity,
    normalize_melody_ending,
    partition_texture_batches,
    prepare_whole_score_live_run,
    resolve_register_enforcement,
    shared_ending_pitch_assignments,
    validate_early_ending,
    validate_material_pitch_range,
    validate_melody_ending,
    validate_whole_score_source_plan,
)


class _FakeRunner:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, str] | None]] = []

    @property
    def call_number(self) -> int:
        return len(self.calls)

    def run(self, step_id: str, prompt: str, input_hashes=None):
        self.calls.append((step_id, prompt, input_hashes))
        return {
            "composition_source": self.responses.pop(0),
            "intent_summary": "fixture",
        }

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    PROJECT_ROOT
    / ".appendix/reference-variance-smoke-v7/runs/reference-a-candidate-2/outputs/piece-plan.dsl"
)


def _plan():
    return parse_piece_plan(PLAN_PATH.read_text(encoding="utf-8"))


def _tonal_prompt_target(status: str = "specified") -> dict[str, object]:
    return {
        "target_version": "reference-generation-target-v6",
        "controls": {"brightness": 0.0},
        "semantic_targets": {
            "piece_plan": [
                {
                    "id": "tonal_hierarchy",
                    "status": status,
                    "scale_policy": "dorian",
                }
            ],
            "score_spec": [
                {
                    "id": "register_envelope",
                    "policy_id": "melody-containing-reference-span-v2",
                },
                {"id": "velocity_shape"},
            ],
        },
    }


def _install_long_form_intent(run_dir: Path) -> dict[str, object]:
    target_path = run_dir / "inputs/prompt-target.json"
    target = json.loads(target_path.read_text(encoding="utf-8"))
    target = merge_generation_intent(
        target,
        {
            "schema_version": 1,
            "decisions": [
                {
                    "decision_id": "climax",
                    "statement": "後半に山場を作る",
                    "decision_kind": "long_form_climax",
                    "stage_targets": [
                        {
                            "generation_stage": stage,
                            "instructions": [f"{stage}で山場を作る"],
                            "observable": {
                                "observable_id": f"climax_{stage}",
                                "scope": "whole_piece",
                                "operator": "shape",
                            },
                            "verification": "listening_only",
                        }
                        for stage in (
                            "piece_plan",
                            "harmonic_skeleton",
                            "melody_collection",
                            "texture_collection",
                            "performance_spec",
                        )
                    ],
                }
            ],
        },
    )
    target_path.write_text(
        json.dumps(target, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def test_harmony_and_melody_contexts_receive_the_same_specified_tonal_target() -> None:
    prompt_target = _tonal_prompt_target()
    attack_frequency = {
        "target": 2.0,
        "candidate_minimum": 1.0,
        "candidate_maximum": 3.0,
        "group_tolerance_ms": 30,
    }

    harmonic = _harmonic_stage_context(_plan(), prompt_target, attack_frequency)
    melody = _melody_semantic_targets(prompt_target)

    assert harmonic["semantic_targets"] == melody[:1]
    assert harmonic["semantic_targets"][0]["id"] == "tonal_hierarchy"
    assert [item["id"] for item in melody] == ["tonal_hierarchy", "register_envelope"]


def test_unverified_reference_brightness_is_not_sent_as_a_tonal_instruction() -> None:
    prompt_target = _tonal_prompt_target("unverified_continuous_reference")
    attack_frequency = {
        "target": 2.0,
        "candidate_minimum": 1.0,
        "candidate_maximum": 3.0,
        "group_tolerance_ms": 30,
    }

    assert _harmonic_stage_context(_plan(), prompt_target, attack_frequency)[
        "semantic_targets"
    ] == []
    assert [item["id"] for item in _melody_semantic_targets(prompt_target)] == [
        "register_envelope"
    ]


@pytest.mark.parametrize(
    ("saved_version", "expected_ids", "register_has_policy"),
    [
        ("reference-generation-target-v2", set(), False),
        ("reference-generation-target-v3", set(), False),
        ("reference-generation-target-v4", {"register_envelope", "velocity_shape"}, False),
        ("reference-generation-target-v5", {"register_envelope", "velocity_shape"}, True),
    ],
)
def test_prompt_target_downgrade_removes_v6_tonal_target(
    saved_version: str,
    expected_ids: set[str],
    register_has_policy: bool,
) -> None:
    downgraded = _prompt_target_for_saved_version(_tonal_prompt_target(), saved_version)
    semantic = downgraded.get("semantic_targets", {})
    ids = {item["id"] for items in semantic.values() for item in items}

    assert downgraded["target_version"] == saved_version
    assert "tonal_hierarchy" not in ids
    assert "piece_plan" not in semantic
    assert ids == expected_ids
    register = next(
        (
            item
            for items in semantic.values()
            for item in items
            if item["id"] == "register_envelope"
        ),
        None,
    )
    assert (register is not None and "policy_id" in register) is register_has_policy


def test_prompt_target_v1_downgrade_still_removes_all_semantic_targets() -> None:
    downgraded = _prompt_target_for_saved_version(
        _tonal_prompt_target(), "reference-generation-target-v1"
    )

    assert "semantic_targets" not in downgraded


def test_prompt_target_v6_downgrade_preserves_explicit_control_and_legacy_semantics() -> None:
    reference_dir = PROJECT_ROOT / ".appendix/reference-profile-v1"
    control_dir = PROJECT_ROOT / ".appendix/control-reference-baseline-v3"
    resolved = json.loads(
        (
            PROJECT_ROOT
            / ".appendix/brightness-middle-normal-generation-v1/inputs/resolved-request.json"
        ).read_text(encoding="utf-8")
    )
    resolved["controls"]["attack_frequency"] = {
        "label": "発音頻度",
        "value": 0.0,
    }
    current = build_reference_generation_target(
        resolved,
        reference_dir=reference_dir,
        control_dir=control_dir,
    ).prompt_target
    legacy = build_reference_generation_target(
        resolved,
        reference_dir=reference_dir,
        control_dir=control_dir,
        legacy_reference_attack_frequency=True,
    ).prompt_target

    downgraded = _prompt_target_for_saved_version(
        current,
        "reference-generation-target-v6",
        legacy_attack_frequency_target=legacy,
    )

    assert downgraded["controls"]["attack_frequency"] == 0.0
    frequency = next(
        item
        for items in downgraded["semantic_targets"].values()
        for item in items
        if item["id"] == "attack_frequency"
    )
    legacy_frequency = next(
        item
        for items in legacy["semantic_targets"].values()
        for item in items
        if item["id"] == "attack_frequency"
    )
    assert frequency == legacy_frequency
    assert frequency["anchor"]["normalized"] != 0.0


def test_explicit_frequency_defers_rendered_group_mismatch_to_candidate_evaluation() -> None:
    rendered_measurement = {
        "frequency_matches_budget": False,
        "texture_shape_matches_budget": True,
        "matches_budget": False,
    }

    assert _rendered_texture_budget_passes(
        {"source": "explicit_corpus_min_max_v1"},
        rendered_measurement,
    ) is True
    assert _rendered_texture_budget_passes(
        {"source": "reference_neighborhood_v3"},
        rendered_measurement,
    ) is False


def _harmonic_material(length: int, root: int = 2, quality: str = "major"):
    half = length // 2
    return WholeHarmonicMaterialDraftV0(
        length,
        HarmonicDraft(
            (
                HarmonicEventDraft(0, half, root, quality),
                HarmonicEventDraft(half, length - half, root, quality),
            )
        ),
    )


def _skeleton(length: int = 48):
    plan = _plan()
    drafts = [_harmonic_material(length) for _ in range(8)]
    drafts[-1] = WholeHarmonicMaterialDraftV0(
        length,
        HarmonicDraft((HarmonicEventDraft(0, length, 2, "major"),)),
    )
    skeleton = assemble_whole_score_skeleton(
        "live-test",
        plan,
        tuple((item.length_units, item.draft) for item in drafts),
    )
    return plan, skeleton


def _v25_capacity_plan() -> PiecePlan:
    return PiecePlan(
        "v25-capacity-test",
        "capacity test",
        2,
        "minor",
        "root",
        "tonic",
        (
            PlanNode("root", None, 0, "whole"),
            PlanNode(
                "a1", "root", 0, "statement",
                duration_weight=14, score_material_id="material_a",
            ),
            PlanNode(
                "b1", "root", 1, "variation",
                duration_weight=12, score_material_id="material_b",
            ),
            PlanNode(
                "c1", "root", 2, "contrast",
                duration_weight=13, score_material_id="material_c",
            ),
            PlanNode(
                "d1", "root", 3, "climax",
                duration_weight=11, score_material_id="material_d",
            ),
            PlanNode(
                "a2", "root", 4, "return", derived_from="a1",
                duration_weight=12, score_material_id="material_a",
            ),
            PlanNode(
                "b2", "root", 5, "variation",
                duration_weight=10, score_material_id="material_b",
            ),
            PlanNode(
                "release", "root", 6, "release",
                duration_weight=8, score_material_id="material_release",
            ),
        ),
    )


def _v25_capacity_materials() -> tuple[WholeHarmonicMaterialDraftV0, ...]:
    materials = tuple(
        _harmonic_material(length, quality="minor")
        for length in (72, 60, 72, 60)
    )
    return (
        *materials,
        WholeHarmonicMaterialDraftV0(
            48,
            HarmonicDraft((HarmonicEventDraft(0, 48, 2, "minor"),)),
        ),
    )


def test_harmonic_capacity_normalization_matches_v25_counterexample() -> None:
    plan = _v25_capacity_plan()
    source = _v25_capacity_materials()

    normalized, diagnostic = normalize_harmonic_capacity(
        plan,
        source,
        minimum_attack_group_count=461,
    )

    assert diagnostic["status"] == "normalized"
    assert diagnostic["source_capacity"]["maximum_attack_group_count"] == 440
    assert diagnostic["effective_capacity"]["maximum_attack_group_count"] == 461
    assert [item.length_units for item in normalized] == [76, 63, 75, 63, 50]
    assert diagnostic["content_invariants"] == {
        "event_count_unchanged": True,
        "event_order_unchanged": True,
        "root_pitch_class_unchanged": True,
        "quality_unchanged": True,
    }
    assert diagnostic["effective_early_ending"]["status"] == "pass"
    assert diagnostic["minimum_expansion_factor"] > 1
    assert diagnostic["maximum_expansion_factor"] < 1.06


def test_harmonic_capacity_normalization_records_unbounded_grid_expansion() -> None:
    plan = _v25_capacity_plan()
    source = (
        *(
            WholeHarmonicMaterialDraftV0(
                5,
                HarmonicDraft(
                    tuple(
                        HarmonicEventDraft(index, 1, (2 + index) % 12, "minor")
                        for index in range(5)
                    )
                ),
            )
            for _ in range(4)
        ),
        WholeHarmonicMaterialDraftV0(
            1,
            HarmonicDraft((HarmonicEventDraft(0, 1, 2, "minor"),)),
        ),
    )

    normalized, diagnostic = normalize_harmonic_capacity(
        plan,
        source,
        minimum_attack_group_count=461,
    )

    assert diagnostic["status"] == "normalized"
    assert diagnostic["source_capacity"]["maximum_attack_group_count"] == 31
    assert diagnostic["effective_capacity"]["maximum_attack_group_count"] >= 461
    assert diagnostic["maximum_expansion_factor"] >= 15
    assert all(diagnostic["content_invariants"].values())
    assert all(
        item.length_units > before.length_units
        for item, before in zip(normalized, source, strict=True)
    )


def test_capacity_expands_reused_material_score_onsets_and_reports_note_events() -> None:
    plan, skeleton = _skeleton(48)

    result = maximum_attack_group_capacity(plan, skeleton, duration_ms=180_000)

    assert result["occurrence_count"] == 11
    assert result["maximum_attack_group_count"] == 523
    assert result["maximum_note_event_count"] == 528
    assert result["maximum_attack_groups_per_second"] == pytest.approx(523 / 180)
    assert result["basis"] == "ending_reserved_integer_score_onset_count"
    assert result["per_material_attack_group_capacity"]["material_8"] == 43
    assert result["per_material_reserved_ending_units"]["material_8"] == 5


def test_capacity_does_not_use_score_onsets_as_rolled_notes_can_split_groups() -> None:
    score_onsets = {0}
    rendered_attack_times_ms = (0, 45)

    assert len(score_onsets) == 1
    assert len(rendered_attack_times_ms) == 2
    assert rendered_attack_times_ms[1] - rendered_attack_times_ms[0] > 30


def test_early_ending_requires_single_tonic_mode_and_two_second_capacity() -> None:
    plan, skeleton = _skeleton(48)

    result = validate_early_ending(plan, skeleton, minimum_hold_ms=2_000)

    assert result["status"] == "pass"
    assert result["final_material_id"] == "material_8"
    assert result["nominal_capacity_ms"] > 2_000

    wrong_harmony = replace(
        skeleton.materials[-1].harmonies[0],
        root_pitch_class=3,
    )
    bad_material = replace(skeleton.materials[-1], harmonies=(wrong_harmony,))
    bad_skeleton = replace(
        skeleton,
        materials=(*skeleton.materials[:-1], bad_material),
    )
    with pytest.raises(WholeScoreLiveRunError, match="tonic"):
        validate_early_ending(plan, bad_skeleton, minimum_hold_ms=2_000)


def test_early_ending_rejects_insufficient_nominal_capacity() -> None:
    plan, skeleton = _skeleton(48)
    short = replace(skeleton.materials[-1], length_units=1)
    short_harmony = replace(short.harmonies[0], duration_units=1)
    short = replace(short, harmonies=(short_harmony,))
    short_skeleton = replace(skeleton, materials=(*skeleton.materials[:-1], short))

    with pytest.raises(WholeScoreLiveRunError, match="hold capacity"):
        validate_early_ending(plan, short_skeleton, minimum_hold_ms=2_000)


def test_prepare_revalidates_provenance_and_separates_generation_from_evaluation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "live"

    result = prepare_whole_score_live_run(PROJECT_ROOT, run_dir)

    assert result["status"] == "prepared"
    assert result["protocol_id"] == "whole-score-staged-generation-live-v6"
    assert result["source_run_status"] == "completed"
    assert result["prompt_target_rebuilt"] is True
    spec = __import__("json").loads((run_dir / "run-spec.json").read_text(encoding="utf-8"))
    assert set(spec["input_hashes"]) == {"generation_inputs", "evaluation_inputs"}
    assert (
        "src/llm_musical_composer/reference_generation_target.py"
        in spec["implementation_hashes"]
    )
    assert "src/llm_musical_composer/texture_budget.py" in spec["implementation_hashes"]
    generation_text = __import__("json").dumps(
        spec["input_hashes"]["generation_inputs"], sort_keys=True
    )
    evaluation_text = __import__("json").dumps(
        spec["input_hashes"]["evaluation_inputs"], sort_keys=True
    )
    assert "score-spec" not in generation_text
    assert "performance-spec" not in generation_text
    assert "final.mid" not in generation_text
    assert "final.mid" in evaluation_text
    assert (run_dir / "inputs/piece-plan.dsl").is_file()
    assert (run_dir / "inputs/prompt-target.json").is_file()
    prompt_target = json.loads(
        (run_dir / "inputs/prompt-target.json").read_text(encoding="utf-8")
    )
    assert prompt_target["target_version"] == "reference-generation-target-v7"
    assert sum(len(items) for items in prompt_target["stage_targets"].values()) == 17
    assert "semantic_targets" in prompt_target
    attack_target = next(
        item
        for item in prompt_target["semantic_targets"]["score_spec"]
        if item["id"] == "attack_texture"
    )
    assert attack_target["maximum_group_size"] == {"anchor": 8}
    assert not (run_dir / "inputs/score-spec.dsl").exists()
    assert spec["register_enforcement"] == {
        "requested_mode": "hard",
        "effective_mode": "hard",
        "promotion_blocker": False,
    }


def _failed_v5_placement() -> PianoTexturePlacementResultV5:
    return PianoTexturePlacementResultV5(
        status="constraint_unplaceable",
        material_id="body",
        notes=(),
        failed_onset=0,
        reason="fixture",
        low_spacing_violations=0,
        rearticulations=(),
        candidate_evaluation_count=1,
        backtrack_count=0,
        zone_projections=(),
        primary_status="constraint_unplaceable",
        primary_failed_onset=0,
        primary_reason="fixture",
        primary_candidate_evaluation_count=1,
        primary_backtrack_count=0,
        fallback_status=None,
        fallback_candidate_evaluation_count=None,
        fallback_backtrack_count=None,
    )


def _failed_v6_placement() -> PianoTexturePlacementResultV6:
    v5 = _failed_v5_placement()
    return PianoTexturePlacementResultV6(
        **v5.__dict__,
        v6_zone_projections=(),
        v5_status=v5.status,
        v5_candidate_evaluation_count=v5.candidate_evaluation_count,
        v5_backtrack_count=v5.backtrack_count,
        v6_fallback_status=None,
        v6_fallback_candidate_evaluation_count=None,
        v6_fallback_backtrack_count=None,
    )


def _failed_v7_placement() -> PianoTexturePlacementResultV7:
    v6 = _failed_v6_placement()
    return PianoTexturePlacementResultV7(
        **v6.__dict__,
        v7_zone_projections=(),
        v6_status=v6.status,
        v6_candidate_evaluation_count=v6.candidate_evaluation_count,
        v6_backtrack_count=v6.backtrack_count,
        v7_fallback_status=None,
        v7_fallback_candidate_evaluation_count=None,
        v7_fallback_backtrack_count=None,
    )


def _failed_v8_placement() -> PianoTexturePlacementResultV8:
    v7 = _failed_v7_placement()
    return PianoTexturePlacementResultV8(
        **v7.__dict__,
        v8_zone_projections=(),
        v7_status=v7.status,
        v7_candidate_evaluation_count=v7.candidate_evaluation_count,
        v7_backtrack_count=v7.backtrack_count,
        v8_fallback_status=None,
        v8_fallback_candidate_evaluation_count=None,
        v8_fallback_backtrack_count=None,
        total_candidate_evaluation_count=1,
        maximum_total_candidate_evaluations=300_000,
    )


@pytest.mark.parametrize(
    ("placement", "expected_schema", "has_placement"),
    (
        (_failed_v5_placement(), 1, False),
        (_failed_v6_placement(), 2, True),
        (_failed_v7_placement(), 3, True),
        (_failed_v8_placement(), 4, True),
    ),
)
def test_stage_failure_changes_schema_for_versioned_placement(
    tmp_path: Path,
    placement: (
        PianoTexturePlacementResultV5
        | PianoTexturePlacementResultV6
        | PianoTexturePlacementResultV7
        | PianoTexturePlacementResultV8
    ),
    expected_schema: int,
    has_placement: bool,
) -> None:
    run_dir = tmp_path / "run"
    RunStore(run_dir, max_calls=1).initialize(
        {
            "schema_version": 1,
            "model_config": {"maximum_external_calls": 1},
        }
    )
    error = _TextureBatchPlacementError("body", placement)

    _save_stage_failure(run_dir, "texture-batch", "fixture", error, {})

    failure = json.loads(
        (run_dir / "failures/texture-batch.json").read_text(encoding="utf-8")
    )
    assert failure["schema_version"] == expected_schema
    assert ("placement" in failure) is has_placement
    if has_placement:
        assert failure["placement"]["status"] == "constraint_unplaceable"


def test_prepare_rejects_register_experiment_without_fixed_provenance(
    tmp_path: Path,
) -> None:
    with pytest.raises(WholeScoreLiveRunError, match="register exception provenance"):
        prepare_whole_score_live_run(
            PROJECT_ROOT,
            tmp_path / "live",
            source=WholeScorePreparedSource(
                run_root=PLAN_PATH.parents[1],
                register_enforcement_mode=(
                    "diagnostic_only_lower_bound_experiment"
                ),
            ),
        )


def test_register_enforcement_defaults_to_hard_and_limits_legacy_prefix() -> None:
    assert resolve_register_enforcement(
        requested_mode="hard",
        target_version="reference-generation-target-v4",
        has_prepared_prefix=True,
        has_verified_legacy_provenance=False,
    )["effective_mode"] == "hard"
    assert resolve_register_enforcement(
        requested_mode="legacy_prefix_diagnostic_only",
        target_version="reference-generation-target-v3",
        has_prepared_prefix=True,
        has_verified_legacy_provenance=True,
    )["effective_mode"] == "legacy_prefix_diagnostic_only"
    with pytest.raises(WholeScoreLiveRunError, match="legacy prefix"):
        resolve_register_enforcement(
            requested_mode="legacy_prefix_diagnostic_only",
            target_version="reference-generation-target-v4",
            has_prepared_prefix=True,
            has_verified_legacy_provenance=True,
        )


def test_register_experiment_counterfactual_requires_normal_failure_and_lower_pass(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    plan, skeleton = _skeleton(48)
    melody_drafts, _ = normalize_melody_ending(
        plan,
        skeleton,
        tuple(_melody((66, 69, 66, 62)) for _ in skeleton.materials),
    )
    melodies = assemble_whole_score_melodies(
        "register-experiment-test",
        plan,
        skeleton,
        melody_drafts,
    )
    target = next(
        item
        for item in json.loads(
            (run_dir / "inputs/prompt-target.json").read_text(encoding="utf-8")
        )["semantic_targets"]["score_spec"]
        if item["id"] == "attack_texture"
    )
    budget = allocate_texture_budget(
        plan,
        skeleton,
        melodies,
        target,
        minimum_attack_group_count=314,
    )
    drafts = _budget_textures(plan, skeleton, melodies)

    result = build_register_exception_counterfactual(
        plan,
        skeleton,
        melodies,
        budget,
        dict(
            zip(
                (item.material_id for item in skeleton.materials),
                drafts,
                strict=True,
            )
        ),
        normal_allowed_pitch_range=(48, 96),
        experimental_allowed_pitch_range=(21, 96),
    )

    assert result["normal"]["status"] == "failed"
    assert result["normal"]["failure"]["kind"] == "validation"
    assert "event feasibility" in result["normal"]["failure"]["detail"]
    assert result["experimental"]["status"] == "passed"


def _register_experiment_fixed_inputs() -> dict[str, Path | dict]:
    v10_root = PROJECT_ROOT / (
        ".appendix/supported-normal-generation-v10-velocity-register-"
        "transition-contract"
    )
    v11_root = PROJECT_ROOT / (
        ".appendix/supported-normal-generation-v11-capacity-repair-"
        "velocity-register"
    )
    v12_root = PROJECT_ROOT / (
        ".appendix/supported-normal-generation-v12-texture-batches-400"
    )
    prompt_target_path = v10_root / "runs/piece-plan/inputs/prompt-target.json"
    return {
        "plan_path": v10_root / "runs/piece-plan/outputs/piece-plan.dsl",
        "prompt_target": json.loads(prompt_target_path.read_text(encoding="utf-8")),
        "harmonic_collection_path": v11_root
        / "inputs/repaired-harmonic-collection.dsl",
        "melody_collection_path": v11_root
        / "runs/whole-score/outputs/melody-collection-main.dsl",
        "texture_budget_path": v11_root
        / "runs/whole-score/outputs/texture-budget.json",
        "failure_path": v12_root
        / "runs/whole-score/failures/texture-collection-batch-001.json",
    }


def test_register_exception_provenance_separates_entry_pass_from_batch_failure() -> None:
    fixed = _register_experiment_fixed_inputs()

    result = build_register_exception_provenance(
        PROJECT_ROOT,
        policy_id=REGISTER_LOWER_BOUND_EXPERIMENT_POLICY_ID,
        texture_batch_maximum_event_count=400,
        **fixed,
    )

    assert result["policy_id"] == REGISTER_LOWER_BOUND_EXPERIMENT_POLICY_ID
    assert result["entry_counterfactual"]["normal"]["status"] == "failed"
    assert result["entry_counterfactual"]["experimental"]["status"] == "passed"
    failure = result["full_batch_counterfactual"]["experimental"]["failure"]
    assert failure["kind"] == "placement"
    assert failure["material_id"] == "material_2"
    assert failure["placement"]["status"] == "search_unplaceable"
    assert failure["placement"]["failed_onset"] == 79
    assert failure["placement"]["candidate_evaluation_count"] == 2295
    assert json.loads(json.dumps(result)) == result


def test_v5_places_the_fixed_v13_lower_foreground_counterexample() -> None:
    v10_root = PROJECT_ROOT / (
        ".appendix/supported-normal-generation-v10-velocity-register-"
        "transition-contract"
    )
    v11_root = PROJECT_ROOT / (
        ".appendix/supported-normal-generation-v11-capacity-repair-"
        "velocity-register"
    )
    v13_root = PROJECT_ROOT / (
        ".appendix/supported-normal-generation-v13-register-lower-bound-experiment"
    )
    plan = parse_piece_plan(
        (v10_root / "runs/piece-plan/outputs/piece-plan.dsl").read_text(
            encoding="utf-8"
        )
    )
    harmonies = parse_harmonic_collection(
        (v11_root / "inputs/repaired-harmonic-collection.dsl").read_text(
            encoding="utf-8"
        )
    )
    melodies = parse_melody_collection(
        (v11_root / "runs/whole-score/outputs/melody-collection-main.dsl").read_text(
            encoding="utf-8"
        )
    )
    failure = json.loads(
        (
            v13_root
            / "runs/whole-score/failures/texture-collection-batch-002.json"
        ).read_text(encoding="utf-8")
    )
    draft = parse_texture_collection(failure["composition_source"])[0]
    skeleton = assemble_whole_score_skeleton(
        "v13-v5-test",
        plan,
        tuple((item.length_units, item.draft) for item in harmonies),
    )
    payload = assemble_whole_score_melodies(
        "v13-v5-test", plan, skeleton, melodies
    )
    score = apply_harmonic_skeleton(plan, payload, skeleton)
    material = next(item for item in skeleton.materials if item.material_id == "material_3")
    target = next(item for item in score.materials if item.material_id == "material_3")
    melody = next(item for item in payload.materials if item.material_id == "material_3")
    budget = json.loads(
        (v11_root / "runs/whole-score/outputs/texture-budget.json").read_text(
            encoding="utf-8"
        )
    )
    material_budget = next(
        item for item in budget["materials"] if item["material_id"] == "material_3"
    )

    v4, _ = assemble_texture(
        "v13-v4-test",
        plan,
        score,
        target,
        material.harmonies,
        melody.notes,
        draft,
        maximum_event_count=material_budget["maximum_texture_event_count"],
        placement_policy="bounded-backtracking-v4",
        allowed_pitch_range=(38, 86),
    )
    v5, scored = assemble_texture(
        "v13-v5-test",
        plan,
        score,
        target,
        material.harmonies,
        melody.notes,
        draft,
        maximum_event_count=material_budget["maximum_texture_event_count"],
        placement_policy="low-foreground-outward-v5",
        allowed_pitch_range=(38, 86),
    )

    assert v4.status == "search_unplaceable"
    assert v5.status == "placed"
    assert v5.primary_status == "search_unplaceable"
    assert v5.fallback_status == "placed"
    assert len(v5.zone_projections) == 38
    assert len(v5.notes) == len(draft.events) == 242
    assert all(item.original_zone == "middle" for item in v5.zone_projections)
    assert all(item.effective_zone == "high" for item in v5.zone_projections)
    assert full_low_spacing_violations(scored) == 0


def test_v6_places_only_the_fixed_v14_upper_foreground_counterexample() -> None:
    root = PROJECT_ROOT / (
        ".appendix/supported-normal-generation-v14-melody-containing-register-v5"
        "/runs/whole-score"
    )
    fixed = {
        root / "inputs/piece-plan.dsl": (
            "425b36623c78453cd462f5b9ef4795ef32e45745bf6ef44ba88af58d3db2e77d"
        ),
        root / "inputs/prepared-harmonic-collection.dsl": (
            "781dfa6ebc7834c012d4ff0fe84785f867dc52c03f25b9b6d5eed97b0ba345a5"
        ),
        root / "inputs/prepared-melody-collection.dsl": (
            "6c8b5d0987d7966e2beb45a0837ed46e7b64fe23d6c7c3b041b2a7ce5a2469fb"
        ),
        root / "inputs/prepared-texture-budget.json": (
            "8906b61a03b6ad6a056364a1cfaf3e37c07f3d434be447f8c9f3e36b03d16341"
        ),
        root / "failures/texture-collection-batch-003.json": (
            "cec8bd782754329df9650c2a4ae84e2cb7e6778dfbd2ccd15177e7cf0698e97d"
        ),
    }
    for path, expected in fixed.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

    plan = parse_piece_plan(
        (root / "inputs/piece-plan.dsl").read_text(encoding="utf-8")
    )
    harmonies = parse_harmonic_collection(
        (root / "inputs/prepared-harmonic-collection.dsl").read_text(
            encoding="utf-8"
        )
    )
    melodies = parse_melody_collection(
        (root / "inputs/prepared-melody-collection.dsl").read_text(
            encoding="utf-8"
        )
    )
    skeleton = assemble_whole_score_skeleton(
        "v14-v6-test",
        plan,
        tuple((item.length_units, item.draft) for item in harmonies),
    )
    payload = assemble_whole_score_melodies(
        "v14-v6-test", plan, skeleton, melodies
    )
    score = apply_harmonic_skeleton(plan, payload, skeleton)
    budget = json.loads(
        (root / "inputs/prepared-texture-budget.json").read_text(encoding="utf-8")
    )
    budgets = {item["material_id"]: item for item in budget["materials"]}
    drafts = dict(
        zip(
            ("material_1", "material_2"),
            parse_texture_collection(
                (root / "outputs/texture-collection-batch-001.dsl").read_text(
                    encoding="utf-8"
                )
            ),
            strict=True,
        )
    )
    drafts["material_3"] = parse_texture_collection(
        (root / "outputs/texture-collection-batch-002.dsl").read_text(
            encoding="utf-8"
        )
    )[0]
    failure = json.loads(
        (root / "failures/texture-collection-batch-003.json").read_text(
            encoding="utf-8"
        )
    )
    drafts.update(
        dict(
            zip(
                ("material_4", "material_5"),
                parse_texture_collection(failure["composition_source"]),
                strict=True,
            )
        )
    )
    results = {}
    for material_id in ("material_1", "material_2", "material_3", "material_4", "material_5"):
        material = next(
            item for item in skeleton.materials if item.material_id == material_id
        )
        target = next(
            item for item in score.materials if item.material_id == material_id
        )
        melody = next(
            item for item in payload.materials if item.material_id == material_id
        )
        arguments = {
            "maximum_event_count": budgets[material_id][
                "maximum_texture_event_count"
            ],
            "allowed_pitch_range": (38, 86),
        }
        v5, _ = assemble_texture(
            "v14-fixed",
            plan,
            score,
            target,
            material.harmonies,
            melody.notes,
            drafts[material_id],
            placement_policy="low-foreground-outward-v5",
            **arguments,
        )
        v6, scored = assemble_texture(
            "v14-fixed",
            plan,
            score,
            target,
            material.harmonies,
            melody.notes,
            drafts[material_id],
            placement_policy="bidirectional-low-spacing-v6",
            **arguments,
        )
        results[material_id] = (v5, v6, scored)

    first_three_event_count = sum(
        len(drafts[item].events)
        for item in ("material_1", "material_2", "material_3")
    )
    assert first_three_event_count == 594
    for material_id in ("material_1", "material_2", "material_3", "material_5"):
        v5, v6, scored = results[material_id]
        assert v5.status == v6.status == "placed"
        assert v6.notes == v5.notes
        assert v6.zone_projections == v5.zone_projections
        assert v6.v6_zone_projections == ()
        assert full_low_spacing_violations(scored) == 0

    v5, v6, scored = results["material_4"]
    assert v5.status == "constraint_unplaceable"
    assert v6.status == "placed"
    assert len(v6.v6_zone_projections) == 21
    assert len(v6.notes) == len(drafts["material_4"].events) == 189
    assert (min(item.pitch for item in v6.notes), max(item.pitch for item in v6.notes)) == (38, 69)
    assert full_low_spacing_violations(scored) == 0

    _, material_5, _ = results["material_5"]
    assert len(material_5.notes) == len(drafts["material_5"].events) == 193
    material_5_range = (
        min(item.pitch for item in material_5.notes),
        max(item.pitch for item in material_5.notes),
    )
    assert material_5_range == (38, 67)


def test_register_exception_provenance_rejects_policy_anchor_mismatch() -> None:
    fixed = _register_experiment_fixed_inputs()
    prompt_target = dict(fixed["prompt_target"])
    prompt_target["target_version"] = "reference-generation-target-v999"

    with pytest.raises(WholeScoreLiveRunError, match="policy input SHA-256"):
        build_register_exception_provenance(
            PROJECT_ROOT,
            policy_id=REGISTER_LOWER_BOUND_EXPERIMENT_POLICY_ID,
            texture_batch_maximum_event_count=400,
            **{**fixed, "prompt_target": prompt_target},
        )


def test_register_exception_provenance_rejects_unknown_policy() -> None:
    fixed = _register_experiment_fixed_inputs()

    with pytest.raises(WholeScoreLiveRunError, match="policy is unknown"):
        build_register_exception_provenance(
            PROJECT_ROOT,
            policy_id="unknown-register-experiment",
            texture_batch_maximum_event_count=400,
            **fixed,
        )


def test_register_exception_provenance_rejects_policy_batch_maximum_mismatch() -> None:
    fixed = _register_experiment_fixed_inputs()

    with pytest.raises(WholeScoreLiveRunError, match="policy batch maximum"):
        build_register_exception_provenance(
            PROJECT_ROOT,
            policy_id=REGISTER_LOWER_BOUND_EXPERIMENT_POLICY_ID,
            texture_batch_maximum_event_count=399,
            **fixed,
        )


def test_register_experiment_is_always_a_promotion_blocker() -> None:
    result = apply_register_promotion_policy(
        True,
        {
            "requested_mode": "diagnostic_only_lower_bound_experiment",
            "effective_mode": "diagnostic_only_lower_bound_experiment",
            "promotion_blocker": True,
        },
    )

    assert result == {
        "base_quality_passes": True,
        "passes": False,
        "promoted": False,
        "status": "completed_unfit",
    }


def test_prepare_accepts_piece_plan_stage_without_source_smf(tmp_path: Path) -> None:
    source_dir = tmp_path / "piece-plan-source"
    (source_dir / "inputs").mkdir(parents=True)
    (source_dir / "outputs").mkdir()
    old_source = PROJECT_ROOT / ".appendix/reference-variance-smoke-v7/runs/reference-a-candidate-2"
    for relative in (
        "inputs/prompt-target.json",
        "inputs/resolved-request.json",
        "outputs/piece-plan.dsl",
        "outputs/piece-plan-quality.json",
        "run-spec.json",
        "run-state.json",
    ):
        target = source_dir / relative
        target.write_bytes((old_source / relative).read_bytes())
    state_path = source_dir / "run-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "running"
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "live"

    result = prepare_whole_score_live_run(
        PROJECT_ROOT,
        run_dir,
        source=WholeScorePreparedSource(
            run_root=source_dir,
            use_legacy_v7_frequency=True,
        ),
    )

    assert result["status"] == "prepared"
    assert result["source_run_status"] == "running"
    spec = json.loads((run_dir / "run-spec.json").read_text(encoding="utf-8"))
    assert set(spec["input_hashes"]["evaluation_inputs"]) == {
        "reference_profiles",
        "capability_v26_smf",
        "multiscale_v8_smf",
    }
    _allow_any_key_release_target(run_dir)
    runner = _FakeRunner(_all_stage_responses())

    completed = execute_prepared_whole_score_live_run(PROJECT_ROOT, run_dir, runner)

    assert completed["status"] == "completed_unfit"
    assert runner.call_number == 5
    assert (run_dir / "staged/final.mid").is_file()
    assert (run_dir / "staged/final.musicxml").is_file()


def test_prepare_snapshots_texture_prefix_provenance(tmp_path: Path) -> None:
    provenance_path = tmp_path / "texture-prefix-provenance.json"
    provenance_path.write_text(
        json.dumps({"schema_version": 1, "status": "verified"}) + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "live"

    result = prepare_whole_score_live_run(
        PROJECT_ROOT,
        run_dir,
        source=WholeScorePreparedSource(
            run_root=PLAN_PATH.parents[1],
            use_legacy_v7_frequency=True,
            texture_prefix_provenance_path=provenance_path,
        ),
    )

    assert result["status"] == "prepared"
    snapshot = run_dir / "inputs/prepared-texture-prefix-provenance.json"
    assert snapshot.read_bytes() == provenance_path.read_bytes()
    spec = json.loads((run_dir / "run-spec.json").read_text(encoding="utf-8"))
    assert "texture_prefix_provenance" in spec["input_hashes"][
        "generation_inputs"
    ]


def test_source_plan_compatibility_rejects_non_release_final_leaf() -> None:
    plan = _plan()
    final_leaf = [node for node in plan.nodes if node.score_material_id is not None][-1]
    changed = replace(final_leaf, role="development")
    invalid = replace(
        plan,
        nodes=tuple(changed if node.node_id == changed.node_id else node for node in plan.nodes),
    )

    with pytest.raises(WholeScoreLiveRunError, match="final leaf"):
        validate_whole_score_source_plan(invalid)


def test_source_plan_compatibility_rejects_a_reused_transition_material() -> None:
    plan = _plan()
    transition = next(
        node
        for node in plan.nodes
        if node.role == "transition" and node.score_material_id is not None
    )
    ordinary = next(
        node
        for node in plan.nodes
        if node.role not in {"transition", "release"}
        and node.score_material_id is not None
    )
    changed = replace(ordinary, score_material_id=transition.score_material_id)
    invalid = replace(
        plan,
        nodes=tuple(
            changed if node.node_id == changed.node_id else node for node in plan.nodes
        ),
    )

    with pytest.raises(WholeScoreLiveRunError, match="transition material"):
        validate_whole_score_source_plan(invalid)


def test_harmony_stage_generates_all_lengths_checks_capacity_and_ending(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    prompt_target = _install_long_form_intent(run_dir)
    materials = [_harmonic_material(48) for _ in range(8)]
    materials[-1] = WholeHarmonicMaterialDraftV0(
        48,
        HarmonicDraft((HarmonicEventDraft(0, 48, 2, "major"),)),
    )
    runner = _FakeRunner([dump_harmonic_collection(materials)])

    skeleton = execute_harmony_stage(PROJECT_ROOT, run_dir, runner)

    assert len(skeleton.materials) == 8
    assert runner.call_number == 1
    assert '"material_count": 8' in runner.calls[0][1]
    assert '"occurrence_count": 11' in runner.calls[0][1]
    assert "8素材を返してください" in runner.calls[0][1]
    assert "11回の出現ごとではありません" in runner.calls[0][1]
    assert "event_id" not in runner.calls[0][1]
    assert "harmony_id" not in runner.calls[0][1]
    assert "creative_climax_harmonic_skeleton" in runner.calls[0][1]
    assert runner.calls[0][2]["creative_targets"] == creative_targets_sha256(
        prompt_target, "harmonic_skeleton"
    )
    capacity = __import__("json").loads(
        (run_dir / "outputs/attack-capacity.json").read_text(encoding="utf-8")
    )
    assert capacity["maximum_attack_groups_per_second"] >= 1.76465137
    raw_capacity = json.loads(
        (run_dir / "outputs/raw-attack-capacity.json").read_text(encoding="utf-8")
    )
    assert raw_capacity == capacity
    normalization = json.loads(
        (run_dir / "outputs/harmonic-capacity-normalization.json").read_text(
            encoding="utf-8"
        )
    )
    assert normalization["status"] == "unchanged"
    assert normalization["raw_harmonic_collection_sha256"] == normalization[
        "effective_harmonic_collection_sha256"
    ]
    assert (run_dir / "outputs/harmonic-skeleton.dsl").is_file()
    assert (run_dir / "outputs/early-ending.json").is_file()


def test_harmony_stage_normalizes_fresh_capacity_without_retry(tmp_path: Path) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    materials = [_harmonic_material(12) for _ in range(8)]
    materials[-1] = WholeHarmonicMaterialDraftV0(
        12,
        HarmonicDraft((HarmonicEventDraft(0, 12, 2, "major"),)),
    )

    raw_source = dump_harmonic_collection(materials)
    runner = _FakeRunner([raw_source])

    skeleton = execute_harmony_stage(PROJECT_ROOT, run_dir, runner)

    raw_capacity = json.loads(
        (run_dir / "outputs/raw-attack-capacity.json").read_text(encoding="utf-8")
    )
    effective_capacity = json.loads(
        (run_dir / "outputs/attack-capacity.json").read_text(encoding="utf-8")
    )
    normalization = json.loads(
        (run_dir / "outputs/harmonic-capacity-normalization.json").read_text(
            encoding="utf-8"
        )
    )
    step = json.loads(
        (run_dir / "steps/harmonic-collection.json").read_text(encoding="utf-8")
    )
    assert runner.call_number == 1
    assert raw_capacity["maximum_attack_group_count"] < 314
    assert effective_capacity["maximum_attack_group_count"] >= 314
    assert normalization["status"] == "normalized"
    assert normalization["raw_harmonic_collection_sha256"] != normalization[
        "effective_harmonic_collection_sha256"
    ]
    assert (run_dir / "responses/harmonic-collection.dsl").read_text(
        encoding="utf-8"
    ) == raw_source
    assert (run_dir / "outputs/harmonic-collection.dsl").read_text(
        encoding="utf-8"
    ) != raw_source
    assert [item.length_units for item in skeleton.materials] != [12] * 8
    assert step["outputs"]["normalization_status"] == "normalized"
    assert step["outputs"]["raw_capacity"] == raw_capacity
    assert step["outputs"]["effective_capacity"] == effective_capacity
    assert step["outputs"]["raw_harmonic_collection_sha256"] == normalization[
        "raw_harmonic_collection_sha256"
    ]
    assert step["outputs"]["effective_harmonic_collection_sha256"] == normalization[
        "effective_harmonic_collection_sha256"
    ]


def test_harmony_stage_rejects_low_capacity_prepared_input_without_normalizing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    materials = [_harmonic_material(12) for _ in range(8)]
    materials[-1] = WholeHarmonicMaterialDraftV0(
        12,
        HarmonicDraft((HarmonicEventDraft(0, 12, 2, "major"),)),
    )
    prepared = dump_harmonic_collection(materials)
    (run_dir / "inputs/prepared-harmonic-collection.dsl").write_text(
        prepared,
        encoding="utf-8",
    )

    with pytest.raises(WholeScoreLiveRunError, match="insufficient attack capacity"):
        execute_harmony_stage(PROJECT_ROOT, run_dir, _FakeRunner([]))

    raw_capacity = json.loads(
        (run_dir / "outputs/raw-attack-capacity.json").read_text(encoding="utf-8")
    )
    effective_capacity = json.loads(
        (run_dir / "outputs/attack-capacity.json").read_text(encoding="utf-8")
    )
    normalization = json.loads(
        (run_dir / "outputs/harmonic-capacity-normalization.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw_capacity == effective_capacity
    assert normalization["status"] == "rejected_prepared"
    assert not (run_dir / "outputs/harmonic-collection.dsl").exists()


def test_harmony_stage_saves_raw_capacity_before_rejecting_ending(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    materials = [_harmonic_material(48) for _ in range(8)]
    materials[-1] = WholeHarmonicMaterialDraftV0(
        48,
        HarmonicDraft((HarmonicEventDraft(0, 48, 3, "major"),)),
    )

    with pytest.raises(WholeScoreLiveRunError, match="tonic"):
        execute_harmony_stage(
            PROJECT_ROOT,
            run_dir,
            _FakeRunner([dump_harmonic_collection(materials)]),
        )

    assert (run_dir / "outputs/raw-attack-capacity.json").is_file()
    assert not (run_dir / "outputs/harmonic-capacity-normalization.json").exists()


def _melody(pitches: tuple[int, ...], length: int = 48) -> MelodyDraft:
    step = length // len(pitches)
    return MelodyDraft(
        "upper",
        tuple(
            MelodyEventDraft(index * step, step, pitch)
            for index, pitch in enumerate(pitches)
        ),
    )


def test_melody_ending_requires_tonic_at_last_usable_onset() -> None:
    plan, skeleton = _skeleton(48)
    drafts = [_melody((66, 69, 66, 62)) for _ in skeleton.materials]
    normalized, diagnostic = normalize_melody_ending(plan, skeleton, tuple(drafts))
    payload = assemble_whole_score_melodies(
        "ending-validation-test",
        plan,
        skeleton,
        normalized,
    )

    result = validate_melody_ending(plan, skeleton, payload)

    assert result["status"] == "pass"
    assert result["final_attack_units"] == 42
    assert result["minimum_final_duration_units"] == 6
    assert result["reserved_ending_units"] == 5
    assert diagnostic["status"] == "normalized"
    assert diagnostic["source_attack_units"] == 36
    assert diagnostic["required_final_attack_units"] == 42
    assert diagnostic["source_duration_units"] == 12
    assert diagnostic["required_final_duration_units"] == 6
    assert normalized[-1].events[-1].pitch == drafts[-1].events[-1].pitch
    assert normalized[-1].events[-1].articulations == drafts[-1].events[-1].articulations

    release = payload.materials[-1]
    short_final = replace(release.notes[-1], duration_units=5)
    invalid_payload = replace(
        payload,
        materials=(
            *payload.materials[:-1],
            replace(release, notes=(*release.notes[:-1], short_final)),
        ),
    )
    with pytest.raises(WholeScoreLiveRunError, match="ending melody"):
        validate_melody_ending(plan, skeleton, invalid_payload)


def test_melody_ending_normalization_is_stable_at_the_shared_attack() -> None:
    plan, skeleton = _skeleton(48)
    drafts = [_melody((66, 69, 66, 62)) for _ in skeleton.materials]
    ending = drafts[-1]
    drafts[-1] = replace(
        ending,
        events=(
            *ending.events[:-1],
            replace(ending.events[-1], at_units=42, duration_units=6),
        ),
    )

    normalized, diagnostic = normalize_melody_ending(plan, skeleton, tuple(drafts))

    assert normalized == tuple(drafts)
    assert diagnostic["status"] == "unchanged"
    assert diagnostic["changed_event_count"] == 0


def test_melody_ending_normalization_does_not_trim_a_prior_note() -> None:
    plan, skeleton = _skeleton(48)
    drafts = [_melody((66, 69, 66, 62)) for _ in skeleton.materials]
    ending = drafts[-1]
    drafts[-1] = replace(
        ending,
        events=(
            *ending.events[:-2],
            replace(ending.events[-2], duration_units=20),
            ending.events[-1],
        ),
    )

    with pytest.raises(WholeScoreLiveRunError, match="shared ending attack"):
        normalize_melody_ending(plan, skeleton, tuple(drafts))


def test_shared_ending_pitch_assignment_reproduces_v24_conflict_and_solution() -> None:
    assert shared_ending_pitch_assignments(
        foreground_pitches=(50,),
        foreground_voice="upper",
        root_pitch_class=2,
        allowed_pitch_range=(45, 95),
    ) == ()

    assignments = shared_ending_pitch_assignments(
        foreground_pitches=(62,),
        foreground_voice="upper",
        root_pitch_class=2,
        allowed_pitch_range=(45, 95),
    )

    assert assignments[0]["pitches"] == {"root": 50, "fifth": 57}
    assert assignments[0]["minimum_required_new_accompaniment_attack_count"] == 2


def test_melody_ending_moves_the_shortest_safe_v24_suffix() -> None:
    plan, skeleton = _skeleton(48)
    drafts = [_melody((66, 69, 66, 62)) for _ in skeleton.materials]
    v24_events = tuple(
        MelodyEventDraft(at_units, duration, pitch, articulations)
        for at_units, duration, pitch, articulations in (
            (0, 2, 81, ("accent",)),
            (2, 2, 78, ()),
            (4, 2, 74, ("staccato",)),
            (6, 2, 69, ()),
            (8, 3, 66, ("tenuto",)),
            (11, 2, 62, ()),
            (13, 2, 66, ()),
            (15, 2, 69, ("accent",)),
            (18, 3, 74, ("tenuto",)),
            (21, 2, 78, ("staccato",)),
            (23, 2, 74, ()),
            (25, 3, 69, ()),
            (28, 2, 66, ("tenuto",)),
            (30, 3, 62, ()),
            (33, 2, 57, ()),
            (35, 3, 54, ("accent",)),
            (38, 2, 57, ()),
            (40, 2, 54, ("tenuto",)),
            (42, 6, 50, ("tenuto",)),
        )
    )
    drafts[-1] = MelodyDraft("upper", v24_events)

    normalized, diagnostic = normalize_melody_ending(
        plan,
        skeleton,
        tuple(drafts),
        allowed_pitch_range=(45, 95),
    )

    release = normalized[-1]
    assert [event.pitch for event in release.events[-5:]] == [69, 66, 69, 66, 62]
    assert release.events[-6].pitch == 62
    assert diagnostic["pitch_transposition"]["suffix_start_units"] == 33
    assert diagnostic["pitch_transposition"]["semitones"] == 12
    assert diagnostic["pitch_transposition"]["changed_event_count"] == 5
    assert diagnostic["maximum_representative_leap_after"] == 7
    assert diagnostic["required_final_pitch_assignment"]["pitches"] == {
        "root": 50,
        "fifth": 57,
    }

    payload = assemble_whole_score_melodies(
        "v24-ending-normalization-test", plan, skeleton, normalized
    )
    assert len(payload.materials[-1].notes) == len(v24_events)


def test_melody_ending_excludes_a_suffix_that_overlaps_a_held_pitch() -> None:
    plan, skeleton = _skeleton(48)
    drafts = [_melody((66, 69, 66, 62)) for _ in skeleton.materials]
    source = MelodyDraft(
        "upper",
        (
            MelodyEventDraft(0, 2, 81),
            MelodyEventDraft(2, 2, 78),
            MelodyEventDraft(4, 2, 74),
            MelodyEventDraft(6, 2, 69),
            MelodyEventDraft(8, 3, 66),
            MelodyEventDraft(11, 2, 62),
            MelodyEventDraft(13, 2, 66),
            MelodyEventDraft(15, 2, 69),
            MelodyEventDraft(18, 3, 74),
            MelodyEventDraft(21, 2, 78),
            MelodyEventDraft(23, 2, 74),
            MelodyEventDraft(25, 3, 69),
            MelodyEventDraft(28, 2, 66),
            MelodyEventDraft(30, 6, 69),
            MelodyEventDraft(33, 2, 57),
            MelodyEventDraft(35, 3, 54),
            MelodyEventDraft(38, 2, 57),
            MelodyEventDraft(40, 2, 54),
            MelodyEventDraft(42, 6, 50),
        ),
    )
    drafts[-1] = source

    normalized, diagnostic = normalize_melody_ending(
        plan,
        skeleton,
        tuple(drafts),
        allowed_pitch_range=(45, 95),
    )

    assert diagnostic["pitch_transposition"]["suffix_start_units"] != 33
    assemble_whole_score_melodies(
        "held-pitch-ending-normalization-test", plan, skeleton, normalized
    )


def test_shared_ending_pitch_assignment_supports_lower_foreground() -> None:
    assignments = shared_ending_pitch_assignments(
        foreground_pitches=(50,),
        foreground_voice="lower",
        root_pitch_class=2,
        allowed_pitch_range=(45, 95),
    )

    assert assignments
    assert min(assignments[0]["pitches"].values()) > 50


def test_release_texture_uses_the_shared_ending_attack() -> None:
    plan, skeleton = _skeleton(48)
    valid = TextureDraft(
        (
            TextureEventDraft(1, 40, 2, "third", "middle"),
            TextureEventDraft(1, 42, 6, "root", "high"),
            TextureEventDraft(1, 42, 6, "fifth", "high"),
        )
    )

    result = staged_run._validate_release_texture_ending(
        plan, skeleton, skeleton.materials[-1], valid
    )

    assert result["status"] == "pass"
    assert result["required_final_attack_units"] == 42
    assert result["final_degrees"] == ["fifth", "root"]

    with_third = replace(
        valid,
        events=(
            *valid.events,
            TextureEventDraft(1, 42, 6, "third", "middle"),
        ),
    )
    thick_result = staged_run._validate_release_texture_ending(
        plan, skeleton, skeleton.materials[-1], with_third
    )
    assert thick_result["actual_final_new_accompaniment_attack_count"] == 3

    invalid = replace(
        valid,
        events=tuple(
            replace(event, at_units=41, duration_units=7)
            if event.at_units == 42
            else event
            for event in valid.events
        ),
    )
    with pytest.raises(WholeScoreLiveRunError, match="shared ending attack"):
        staged_run._validate_release_texture_ending(
            plan, skeleton, skeleton.materials[-1], invalid
        )


def test_release_texture_does_not_carry_a_prior_event_over_shared_ending() -> None:
    plan, skeleton = _skeleton(48)
    invalid = TextureDraft(
        (
            TextureEventDraft(1, 40, 3, "third", "middle"),
            TextureEventDraft(1, 42, 6, "root", "high"),
            TextureEventDraft(1, 42, 6, "fifth", "high"),
        )
    )

    with pytest.raises(WholeScoreLiveRunError, match="carries over"):
        staged_run._validate_release_texture_ending(
            plan, skeleton, skeleton.materials[-1], invalid
        )


def _attack_texture_target() -> dict[str, object]:
    return {
        "id": "attack_texture",
        "neighborhood_center": [0.41961779, 0.15084904, 0.11403406, 0.31549911],
        "neighborhood_radius": 0.09450089,
        "notes_per_attack": {
            "neighborhood_center": 2.59314085,
            "neighborhood_radius": 0.27352582,
        },
        "maximum_group_size": {"anchor": 8},
    }


def _allow_any_key_release_target(run_dir: Path) -> None:
    prompt_target_path = run_dir / "inputs/prompt-target.json"
    prompt_target = json.loads(prompt_target_path.read_text(encoding="utf-8"))
    key_held_target = next(
        item
        for item in prompt_target["semantic_targets"]["rendered_surface"]
        if item["id"] == "key_held_texture"
    )
    key_held_target["neighborhood_radius"] = 1.0
    prompt_target_path.write_text(
        json.dumps(prompt_target, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_semantic_texture_fit_requires_attack_and_key_held_targets() -> None:
    semantic = {
        "attack_texture": {
            "within_neighborhood": True,
            "notes_per_attack_within_neighborhood": True,
            "maximum_group_size_within_limit": True,
        },
        "key_held_texture": {"within_neighborhood": True},
    }

    assert _semantic_texture_fit(semantic) is True
    semantic["key_held_texture"]["within_neighborhood"] = False
    assert _semantic_texture_fit(semantic) is False


def _key_release_fixture() -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    plan = PiecePlan(
        "key-release-plan",
        "キー解放校正",
        0,
        "major",
        "root",
        "tonic",
        (
            PlanNode("root", None, 0, "whole"),
            PlanNode(
                "ending",
                "root",
                0,
                "release",
                harmonic_focus=0,
                duration_weight=1,
                score_material_id="ending-material",
            ),
        ),
    )
    score = ScoreSpec(
        "key-release-score",
        4,
        (
            ScoreMaterial(
                "ending-material",
                4,
                (
                    ScoreNote("upper-long", 0, 4, 60, "upper"),
                    ScoreNote("lower-long", 0, 4, 48, "lower"),
                    ScoreNote("upper-ending", 2, 2, 64, "upper"),
                    ScoreNote("lower-short", 2, 1, 52, "lower"),
                ),
            ),
        ),
    )
    performance = PerformanceSpec(
        "key-release-performance",
        4_000,
        64,
        "subtle-v1",
        (NodePerformance("ending", articulation_profile="score"),),
    )
    return plan, score, performance


def test_key_release_calibration_selects_largest_passing_integer_percent() -> None:
    plan, score, performance = _key_release_fixture()
    target = {
        "neighborhood_center": [
            200 / 3_800,
            1_350 / 3_800,
            2_250 / 3_800,
            0.0,
            0.0,
        ],
        "neighborhood_radius": 1e-9,
    }

    calibrated, rendered, report = calibrate_key_release(
        plan,
        score,
        performance,
        target,
        minimum_percent=40,
    )

    assert calibrated.key_release_percent == 50
    assert report["selected_percent"] == 50
    assert report["status"] == "pass"
    assert report["terminal_exempt_note_count"] == 1
    assert report["rounding_policy"] == "single-round-after-articulation-and-percent-v1"
    assert rendered.duration_ms == 4_000


def test_key_release_calibration_rejects_unreachable_search_range() -> None:
    plan, score, performance = _key_release_fixture()
    target = {
        "neighborhood_center": [
            200 / 3_800,
            1_350 / 3_800,
            2_250 / 3_800,
            0.0,
            0.0,
        ],
        "neighborhood_radius": 1e-9,
    }

    with pytest.raises(WholeScoreLiveRunError, match="key release calibration"):
        calibrate_key_release(
            plan,
            score,
            performance,
            target,
            minimum_percent=51,
        )


def test_key_release_calibration_can_return_nearest_unfit_when_opted_in() -> None:
    plan, score, performance = _key_release_fixture()
    target = {
        "neighborhood_center": [
            200 / 3_800,
            1_350 / 3_800,
            2_250 / 3_800,
            0.0,
            0.0,
        ],
        "neighborhood_radius": 1e-9,
    }

    calibrated, rendered, report = calibrate_key_release(
        plan,
        score,
        performance,
        target,
        minimum_percent=51,
        on_unreachable="nearest_unfit",
    )

    assert calibrated.key_release_percent == 51
    assert rendered.duration_ms == 4_000
    assert report["status"] == "nearest_unfit"
    assert report["passes"] is False
    assert report["selected_percent"] == 51
    assert report["radius_excess"] > 0


def test_saved_zero_call_run_can_record_candidate_evaluation_steps(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "zero-call"
    RunStore(run_dir, max_calls=0).initialize(
        {
            "schema_version": 1,
            "model_config": {"maximum_external_calls": 0},
        }
    )

    assert staged_run._run_maximum_external_calls(run_dir) == 0
    staged_run._run_store(run_dir).record_step(
        "rendered-output-validation",
        "completed",
        {},
        {"status": "pass"},
    )

    state = json.loads((run_dir / "run-state.json").read_text(encoding="utf-8"))
    assert state["steps"]["rendered-output-validation"]["status"] == "completed"


def test_default_velocity_calibration_preserves_profile_without_clipping() -> None:
    plan, score, performance = _key_release_fixture()
    performance = replace(
        performance,
        node_performances=(NodePerformance("ending", dynamics_profile="shape"),),
    )
    target = {
        "neighborhood_center": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "neighborhood_radius": 0.0,
    }

    calibrated, rendered, report = calibrate_default_velocity(
        plan, score, performance, target
    )

    assert calibrated.default_velocity == 80
    assert calibrated.node_performances == performance.node_performances
    assert report["clip_count"] == 0
    assert report["before"]["clip_count"] == 0
    assert report["after"]["clip_count"] == 0
    assert report["after"]["distinct_velocity_count"] > 1
    assert {note.velocity for note in rendered.notes} == {80, 84}


def test_default_velocity_calibration_rejects_an_unreachable_distribution() -> None:
    plan, score, performance = _key_release_fixture()
    target = {
        "neighborhood_center": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        "neighborhood_radius": 0.0,
    }

    with pytest.raises(WholeScoreLiveRunError, match="no passing unclipped integer"):
        calibrate_default_velocity(plan, score, performance, target)


def test_default_velocity_calibration_accounts_for_accompaniment_adjustment() -> None:
    plan, score, performance = _key_release_fixture()
    material = replace(
        score.materials[0],
        harmonies=(ScoreHarmony("tonic", 0, 4, 0, "major"),),
        foreground_voice="upper",
    )
    score = replace(score, materials=(material,))
    performance = replace(
        performance,
        default_velocity=5,
        velocity_policy_id="foreground-accompaniment-harmony-shape-v1",
    )
    target = {
        "neighborhood_center": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "neighborhood_radius": 0.0,
    }

    calibrated, rendered, report = calibrate_default_velocity(
        plan, score, performance, target
    )

    assert calibrated.default_velocity == 7
    assert report["before"]["clip_count"] == 2
    assert report["after"]["clip_count"] == 0
    assert {note.velocity for note in rendered.notes if note.voice == "lower"} == {1}


def test_texture_budget_jointly_preserves_frequency_and_attack_size() -> None:
    plan, skeleton = _skeleton(48)
    melodies = assemble_whole_score_melodies(
        "budget-test",
        plan,
        skeleton,
        tuple(_melody((66, 69, 66, 62)) for _ in skeleton.materials),
    )

    result = allocate_texture_budget(
        plan,
        skeleton,
        melodies,
        _attack_texture_target(),
        minimum_attack_group_count=314,
    )

    score = result["score_spec_target"]
    assert score["attack_group_count"] == 314
    assert sum(score["attack_size_counts"].values()) == 314
    assert score["note_event_count"] == 730
    assert score["notes_per_attack"] >= 2.59314085 - 0.27352582
    assert score["distribution_distance"] <= 0.09450089
    assert score["maximum_group_size"] == 8
    assert score["note_event_count"] > maximum_attack_group_capacity(
        plan, skeleton
    )["maximum_note_event_count"]
    materials = {item["material_id"]: item for item in result["materials"]}
    assert materials["material_1"]["occurrence_count"] == 3
    assert materials["material_2"]["occurrence_count"] == 2
    assert materials["material_3"]["occurrence_count"] == 1
    assert sum(
        item["occurrence_count"] * item["combined_attack_group_count"]
        for item in materials.values()
    ) == 314
    assert sum(
        item["occurrence_count"] * item["combined_note_event_count"]
        for item in materials.values()
    ) == 730
    assert all(item["required_texture_event_count"] > 0 for item in materials.values())


def test_melody_stages_generate_non_transitions_before_connected_transitions(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    plan, skeleton = _skeleton(48)
    del plan
    first_pass = (
        _melody((66, 69, 66, 62)),
        _melody((69, 66, 62, 66)),
        _melody((62, 66, 69, 66)),
        _melody((69, 66, 62, 66)),
        _melody((66, 69, 62, 66)),
        _melody((66, 69, 66, 62)),
    )
    transitions = (
        _melody((66, 69, 66, 62)),
        _melody((66, 69, 66, 62)),
    )
    runner = _FakeRunner(
        [dump_melody_collection(first_pass), dump_melody_collection(transitions)]
    )

    payload = execute_melody_stages(PROJECT_ROOT, run_dir, runner, skeleton)

    assert runner.call_number == 2
    assert [call[0] for call in runner.calls] == [
        "melody-collection-main",
        "melody-collection-transition",
    ]
    assert len(payload.materials) == 8
    assert '"register_envelope"' in runner.calls[0][1]
    assert '"allowed_pitch_range"' in runner.calls[1][1]
    assert '"register_progress"' in runner.calls[1][1]
    register_range = json.loads(
        (run_dir / "outputs/register-range.json").read_text(encoding="utf-8")
    )
    assert register_range["policy_id"] == "melody-containing-reference-span-v2"
    assert register_range["minimum_pitch"] <= 62
    assert register_range["maximum_pitch"] >= 69
    assessments = __import__("json").loads(
        (run_dir / "outputs/transition-assessments.json").read_text(encoding="utf-8")
    )
    assert all(item["passes"] for item in assessments["items"])


def test_register_measurement_expands_reused_materials_over_every_leaf() -> None:
    plan, skeleton = _skeleton(48)
    melodies = assemble_whole_score_melodies(
        "register-test",
        plan,
        skeleton,
        tuple(_melody((60, 64, 67, 72)) for _ in skeleton.materials),
    )
    score = ScoreSpec(
        "register-score",
        skeleton.divisions,
        tuple(
            ScoreMaterial(
                item.material_id,
                item.notes[-1].at_units + item.notes[-1].duration_units,
                item.notes,
                foreground_voice=item.foreground_voice,
            )
            for item in melodies.materials
        ),
    )
    pitch_map = {
        item.material_id: [note.pitch for note in item.notes]
        for item in melodies.materials
    }
    target = {
        "target_span_semitones": 12,
        "maximum_span_semitones": 24,
        "pitch_range": {
            "neighborhood_center": [12 / 88],
            "neighborhood_radius": 0.2,
        },
        "relative_register_diagnostic": {
            "neighborhood_center": [0, 0, 0, 1, 0, 0, 0],
            "neighborhood_radius": 1,
        },
    }

    progress = measure_register_progress(plan, pitch_map)
    register_range = build_allowed_pitch_range(plan, pitch_map, target)
    diagnostic = measure_register_diagnostic(plan, score, target)

    assert progress["note_count"] == 44
    assert diagnostic["note_count"] == 44
    assert diagnostic["pitch_span_semitones"] == 12
    assert diagnostic["pitch_range"]["actual"] == [12 / 88]
    assert register_range["maximum_pitch"] - register_range["minimum_pitch"] == 24


def test_transition_pitch_range_validator_rejects_an_expansion() -> None:
    with pytest.raises(WholeScoreLiveRunError, match="shared allowed pitch range"):
        validate_material_pitch_range(
            {"transition": [59, 60, 81]},
            {"minimum_pitch": 48, "maximum_pitch": 72},
        )


def test_v5_register_window_contains_the_confirmed_melody() -> None:
    plan, skeleton = _skeleton(48)
    pitch_map = {item.material_id: [38, 72, 72, 85] for item in skeleton.materials}
    target = {
        "policy_id": "melody-containing-reference-span-v2",
        "target_span_semitones": 47,
        "maximum_span_semitones": 48,
    }

    result = build_allowed_pitch_range(plan, pitch_map, target)

    assert (result["minimum_pitch"], result["maximum_pitch"]) == (38, 86)
    assert result["reachable_after_melody"] is True
    assert result["preferred_pitch_range"] == [48, 96]
    assert result["melody_containing_lower_bound_range"] == [37, 38]
    assert result["lower_bound_projection_semitones"] == -10


def test_melody_stages_skip_transition_call_when_plan_has_none(tmp_path: Path) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    original = _plan()
    nodes = tuple(
        replace(node, role="variation") if node.role == "transition" else node
        for node in original.nodes
    )
    plan = replace(original, nodes=nodes)
    (run_dir / "inputs/piece-plan.dsl").write_text(
        dump_piece_plan(plan), encoding="utf-8"
    )
    drafts = [_harmonic_material(48) for _ in range(8)]
    drafts[-1] = WholeHarmonicMaterialDraftV0(
        48,
        HarmonicDraft((HarmonicEventDraft(0, 48, 2, "major"),)),
    )
    skeleton = assemble_whole_score_skeleton(
        "no-transition-test",
        plan,
        tuple((item.length_units, item.draft) for item in drafts),
    )
    melodies = tuple(_melody((66, 69, 66, 62)) for _ in range(8))
    runner = _FakeRunner([dump_melody_collection(melodies)])

    payload = execute_melody_stages(PROJECT_ROOT, run_dir, runner, skeleton)

    assert runner.call_number == 1
    assert [call[0] for call in runner.calls] == ["melody-collection-main"]
    assert len(payload.materials) == 8
    assessments = json.loads(
        (run_dir / "outputs/transition-assessments.json").read_text(encoding="utf-8")
    )
    assert assessments["items"] == []


def _texture(length: int = 48, harmony_count: int = 2) -> TextureDraft:
    events = []
    segment = length // harmony_count
    per_harmony = 24 // harmony_count
    for harmony_index, start in enumerate(range(0, length, segment)):
        onsets = [start, *(start + offset for offset in range(1, segment, 2))]
        for index, onset in enumerate(onsets[:per_harmony]):
            events.append(
                TextureEventDraft(
                    harmony_index,
                    onset,
                    1,
                    "root" if index % 2 == 0 else "fifth",
                    "bass",
                )
            )
    return TextureDraft(tuple(events))


def _ending_texture(length: int = 48) -> TextureDraft:
    return TextureDraft(
        (
            TextureEventDraft(0, 0, 6, "root", "bass"),
            TextureEventDraft(0, length - 12, 12, "root", "bass"),
            TextureEventDraft(0, length - 12, 12, "fifth", "bass"),
        )
    )


def _budget_textures(
    plan: object,
    skeleton: object,
    payload: object,
) -> tuple[TextureDraft, ...]:
    budget = allocate_texture_budget(
        plan,
        skeleton,
        payload,
        _attack_texture_target(),
        minimum_attack_group_count=314,
    )
    payloads = {item.material_id: item for item in payload.materials}
    drafts = []
    for material, target in zip(skeleton.materials, budget["materials"], strict=True):
        melody = payloads[material.material_id]
        melody_sizes: dict[int, int] = {}
        for note in melody.notes:
            melody_sizes[note.at_units] = melody_sizes.get(note.at_units, 0) + 1
        group_count = target["combined_attack_group_count"]
        onsets = sorted(melody_sizes)
        onsets.extend(
            onset
            for onset in range(material.length_units)
            if onset not in melody_sizes
        )
        onsets = onsets[:group_count]
        sizes = [
            size
            for size, bucket in zip(
                (1, 2, 3, 4),
                ("one", "two", "three", "four_or_more"),
                strict=True,
            )
            for _ in range(target["attack_size_counts"][bucket])
        ]
        sizes.sort(reverse=True)
        assigned = dict(zip(onsets, sizes, strict=True))
        if material.material_id == "material_8" and assigned[max(melody_sizes)] < 3:
            donor = next(onset for onset in onsets if assigned[onset] >= 3)
            assigned[donor], assigned[max(melody_sizes)] = (
                assigned[max(melody_sizes)],
                assigned[donor],
            )
        events = []
        tones = (
            ("root", "bass"),
            ("fifth", "low"),
            ("root", "middle"),
            ("third", "middle"),
        )
        for onset in sorted(onsets):
            harmony_index = max(
                index
                for index, harmony in enumerate(material.harmonies)
                if harmony.at_units <= onset
            )
            needed = assigned[onset] - melody_sizes.get(onset, 0)
            for degree, zone in tones[:needed]:
                duration = (
                    material.length_units - onset
                    if material.material_id == "material_8"
                    and onset == max(melody_sizes)
                    and degree in {"root", "fifth"}
                    else 1
                )
                events.append(
                    TextureEventDraft(
                        harmony_index,
                        onset,
                        duration,
                        degree,
                        zone,
                    )
                )
        assert len(events) == target["required_texture_event_count"]
        drafts.append(TextureDraft(tuple(events)))
    return tuple(drafts)


def test_texture_stage_generates_accompaniment_for_every_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    _, skeleton = _skeleton(48)
    first_pass = (
        _melody((66, 69, 66, 62)),
        _melody((69, 66, 62, 66)),
        _melody((62, 66, 69, 66)),
        _melody((69, 66, 62, 66)),
        _melody((66, 69, 62, 66)),
        _melody((66, 69, 66, 62)),
    )
    transitions = (
        _melody((66, 69, 66, 62)),
        _melody((66, 69, 66, 62)),
    )
    melody_runner = _FakeRunner(
        [dump_melody_collection(first_pass), dump_melody_collection(transitions)]
    )
    payload = execute_melody_stages(PROJECT_ROOT, run_dir, melody_runner, skeleton)
    assert '"required_final_attack_units"' in melody_runner.calls[0][1]
    assert '"required_final_duration_units"' in melody_runner.calls[0][1]
    assert '"required_final_pitch_classes"' in melody_runner.calls[0][1]
    texture_runner = _FakeRunner(
        [dump_texture_collection(_budget_textures(_plan(), skeleton, payload))]
    )
    reachability_calls = []
    reachability = staged_run.texture_budget_onset_capacity_reachability

    def record_reachability(*args, **kwargs):
        reachability_calls.append(kwargs)
        return reachability(*args, **kwargs)

    monkeypatch.setattr(
        staged_run,
        "texture_budget_onset_capacity_reachability",
        record_reachability,
    )

    result = execute_texture_stage(
        PROJECT_ROOT,
        run_dir,
        texture_runner,
        skeleton,
        payload,
    )

    assert texture_runner.call_number == 1
    assert len(result.score.materials) == 8
    assert all(len(material.notes) > 4 for material in result.score.materials)
    assert result.low_spacing_violations == (0,) * 8
    assert (run_dir / "outputs/score-spec.dsl").is_file()
    assert '"required_final_attack_units"' in texture_runner.calls[0][1]
    assert '"required_final_duration_units"' in texture_runner.calls[0][1]
    assert '"required_final_pitch_classes"' in texture_runner.calls[0][1]
    assert '"minimum_required_final_new_accompaniment_attack_count": 2' in (
        texture_runner.calls[0][1]
    )
    assert '"minimum_new_accompaniment_attacks": {' in texture_runner.calls[0][1]
    assert '"scope": "texture_generation_feasibility_v3"' in texture_runner.calls[0][1]
    assert '"policy_id": "melody-led-harmony-start-v1"' in texture_runner.calls[0][1]
    assert "確定旋律が新しい和声を示す" in texture_runner.calls[0][1]
    assert '"required_final_pitch_assignment"' in texture_runner.calls[0][1]
    assert '"occurrence_count": 3' in texture_runner.calls[0][1]
    assert "`occurrence_count`を掛けた合計" in texture_runner.calls[0][1]
    assert '"attack_texture"' in texture_runner.calls[0][1]
    assert '"foreground_accompaniment_coordination"' in texture_runner.calls[0][1]
    assert '"key_held_texture"' in texture_runner.calls[0][1]
    assert '"register_envelope"' in texture_runner.calls[0][1]
    assert '"allowed_pitch_range"' in texture_runner.calls[0][1]
    assert '"register_progress"' in texture_runner.calls[0][1]
    assert '"rhythm_time.attack_size"' not in texture_runner.calls[0][1]
    placement = json.loads(
        (run_dir / "outputs/texture-placement.json").read_text(encoding="utf-8")
    )
    assert placement["schema_version"] == 4
    assert placement["placement_policy"] == "onset-feasible-zone-v7"
    assert all("rearticulations" in item for item in placement["placements"])
    register_diagnostic = json.loads(
        (run_dir / "outputs/register-diagnostic.json").read_text(encoding="utf-8")
    )
    assert register_diagnostic["population"] == "piece_plan_leaf_expanded_score_notes"
    reachability = json.loads(
        (run_dir / "outputs/texture-onset-capacity-reachability.json").read_text(
            encoding="utf-8"
        )
    )
    assert reachability["materials"][-1][
        "minimum_new_accompaniment_attacks"
    ] == {"0": 1, "42": 2}
    assert reachability["materials"][-1]["last_usable_attack_units"] == 42
    assert [call["last_usable_attack_units"] for call in reachability_calls] == [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        42,
    ]


def test_stable_profile_allows_nine_whole_score_calls_and_fixes_policies(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "stable"

    prepare_whole_score_live_run(
        PROJECT_ROOT,
        run_dir,
        source=WholeScorePreparedSource(
            run_root=PLAN_PATH.parents[1],
            use_legacy_v7_frequency=True,
            texture_batch_maximum_event_count=400,
            texture_batch_maximum_count=4,
            maximum_external_calls=9,
            generation_profile_id="stable-staged-v1",
            velocity_policy_id="foreground-accompaniment-harmony-shape-v1",
            key_release_unreachable_policy="nearest_unfit",
        ),
    )

    spec = json.loads((run_dir / "run-spec.json").read_text(encoding="utf-8"))
    assert spec["generation_profile_id"] == "stable-staged-v1"
    assert spec["model_config"]["maximum_external_calls"] == 9
    assert spec["texture_batch_maximum_event_count"] == 400
    assert spec["texture_batch_maximum_count"] == 4
    assert spec["velocity_policy_id"] == (
        "foreground-accompaniment-harmony-shape-v1"
    )
    assert spec["key_release_unreachable_policy"] == "nearest_unfit"
    assert spec["texture_placement_policy"] == "onset-feasible-zone-v7"


def test_stable_v2_profile_fixes_search_aware_v8(tmp_path: Path) -> None:
    run_dir = tmp_path / "stable-v2"

    prepare_whole_score_live_run(
        PROJECT_ROOT,
        run_dir,
        source=WholeScorePreparedSource(
            run_root=PLAN_PATH.parents[1],
            use_legacy_v7_frequency=True,
            texture_batch_maximum_event_count=400,
            texture_batch_maximum_count=4,
            maximum_external_calls=9,
            generation_profile_id="stable-staged-v2",
            velocity_policy_id="foreground-accompaniment-harmony-shape-v1",
            key_release_unreachable_policy="nearest_unfit",
        ),
    )

    spec = json.loads((run_dir / "run-spec.json").read_text(encoding="utf-8"))
    assert spec["generation_profile_id"] == "stable-staged-v2"
    assert spec["texture_placement_policy"] == "search-aware-onset-zone-v8"


def test_prepared_continuation_profile_accepts_explicit_ceilings_and_v8(
    tmp_path: Path,
) -> None:
    prepared_paths = {
        "harmonic_collection_path": tmp_path / "harmonic-collection.dsl",
        "melody_collection_path": tmp_path / "melody-collection.dsl",
        "texture_budget_path": tmp_path / "texture-budget.json",
        "texture_collection_prefix_path": tmp_path / "texture-prefix.dsl",
        "texture_prefix_provenance_path": tmp_path / "texture-prefix-provenance.json",
    }
    for path in prepared_paths.values():
        path.write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "prepared-continuation"

    prepare_whole_score_live_run(
        PROJECT_ROOT,
        run_dir,
        source=WholeScorePreparedSource(
            run_root=PLAN_PATH.parents[1],
            use_legacy_v7_frequency=True,
            texture_batch_maximum_event_count=146,
            texture_batch_maximum_count=2,
            maximum_external_calls=2,
            generation_profile_id="prepared-continuation-v1",
            velocity_policy_id="foreground-accompaniment-harmony-shape-v1",
            key_release_unreachable_policy="nearest_unfit",
            register_enforcement_mode="hard",
            **prepared_paths,
        ),
    )

    spec = json.loads((run_dir / "run-spec.json").read_text(encoding="utf-8"))
    assert spec["generation_profile_id"] == "prepared-continuation-v1"
    assert spec["model_config"]["maximum_external_calls"] == 2
    assert spec["texture_batch_maximum_event_count"] == 146
    assert spec["texture_batch_maximum_count"] == 2
    assert spec["texture_placement_policy"] == "search-aware-onset-zone-v8"


def test_prepared_continuation_profile_requires_all_prepared_inputs(
    tmp_path: Path,
) -> None:
    prepared_harmony = tmp_path / "harmonic-collection.dsl"
    prepared_melody = tmp_path / "melody-collection.dsl"
    prepared_budget = tmp_path / "texture-budget.json"
    prepared_prefix = tmp_path / "texture-prefix.dsl"
    for path in (
        prepared_harmony,
        prepared_melody,
        prepared_budget,
        prepared_prefix,
    ):
        path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        WholeScoreLiveRunError,
        match="prepared continuation profile settings conflict",
    ):
        prepare_whole_score_live_run(
            PROJECT_ROOT,
            tmp_path / "missing-provenance",
            source=WholeScorePreparedSource(
                run_root=PLAN_PATH.parents[1],
                use_legacy_v7_frequency=True,
                harmonic_collection_path=prepared_harmony,
                melody_collection_path=prepared_melody,
                texture_budget_path=prepared_budget,
                texture_collection_prefix_path=prepared_prefix,
                texture_batch_maximum_event_count=146,
                texture_batch_maximum_count=2,
                maximum_external_calls=2,
                generation_profile_id="prepared-continuation-v1",
                velocity_policy_id="foreground-accompaniment-harmony-shape-v1",
                key_release_unreachable_policy="nearest_unfit",
                register_enforcement_mode="hard",
            ),
        )


def test_texture_stage_generates_and_validates_consecutive_batches(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(
        PROJECT_ROOT,
        run_dir,
        source=WholeScorePreparedSource(
            run_root=PLAN_PATH.parents[1],
            use_legacy_v7_frequency=True,
            texture_batch_maximum_event_count=400,
            maximum_external_calls=5,
        ),
    )
    plan, skeleton = _skeleton(48)
    melody_drafts, _ = normalize_melody_ending(
        plan,
        skeleton,
        tuple(_melody((66, 69, 66, 62)) for _ in skeleton.materials),
    )
    payload = assemble_whole_score_melodies(
        "texture-batch-test",
        plan,
        skeleton,
        melody_drafts,
    )
    target = next(
        item
        for item in json.loads(
            (run_dir / "inputs/prompt-target.json").read_text(encoding="utf-8")
        )["semantic_targets"]["score_spec"]
        if item["id"] == "attack_texture"
    )
    budget = allocate_texture_budget(
        plan,
        skeleton,
        payload,
        target,
        minimum_attack_group_count=314,
    )
    batches = partition_texture_batches(
        budget["materials"], maximum_event_count=400
    )
    assert len(batches) > 1
    all_drafts = _budget_textures(plan, skeleton, payload)
    draft_by_material = dict(
        zip(
            (item.material_id for item in skeleton.materials),
            all_drafts,
            strict=True,
        )
    )
    runner = _FakeRunner(
        [
            dump_texture_collection(
                tuple(draft_by_material[item["material_id"]] for item in batch)
            )
            for batch in batches
        ]
    )

    result = execute_texture_stage(
        PROJECT_ROOT,
        run_dir,
        runner,
        skeleton,
        payload,
    )

    assert [call[0] for call in runner.calls] == [
        f"texture-collection-batch-{index:03d}"
        for index in range(1, len(batches) + 1)
    ]
    assert len(result.score.materials) == len(skeleton.materials)
    assert (run_dir / "outputs/texture-collection.dsl").is_file()
    assert (run_dir / "steps/texture-collection.json").is_file()
    for index in range(1, len(batches) + 1):
        validation_path = (
            run_dir
            / f"outputs/texture-collection-batch-{index:03d}-validation.json"
        )
        assert validation_path.is_file()
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        assert validation["schema_version"] == 3
        assert validation["placement_policy"] == "onset-feasible-zone-v7"


def test_texture_stage_separates_register_goal_from_experimental_boundary(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    plan, skeleton = _skeleton(48)
    melody_drafts, _ = normalize_melody_ending(
        plan,
        skeleton,
        tuple(_melody((66, 69, 66, 62)) for _ in skeleton.materials),
    )
    payload = assemble_whole_score_melodies(
        "register-stage-context-test",
        plan,
        skeleton,
        melody_drafts,
    )
    prompt_target = json.loads(
        (run_dir / "inputs/prompt-target.json").read_text(encoding="utf-8")
    )
    register_target = next(
        item
        for item in prompt_target["semantic_targets"]["score_spec"]
        if item["id"] == "register_envelope"
    )
    normal = build_allowed_pitch_range(
        plan,
        {
            material.material_id: [note.pitch for note in material.notes]
            for material in payload.materials
        },
        register_target,
    )
    spec_path = run_dir / "run-spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["register_enforcement"] = {
        "requested_mode": "diagnostic_only_lower_bound_experiment",
        "effective_mode": "diagnostic_only_lower_bound_experiment",
        "promotion_blocker": True,
        "normal_allowed_pitch_range": [
            normal["minimum_pitch"],
            normal["maximum_pitch"],
        ],
        "experimental_allowed_pitch_range": [21, normal["maximum_pitch"]],
        "provenance_sha256": "fixture",
    }
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runner = _FakeRunner(
        [dump_texture_collection(_budget_textures(plan, skeleton, payload))]
    )

    execute_texture_stage(PROJECT_ROOT, run_dir, runner, skeleton, payload)

    prompt = runner.calls[0][1]
    assert '"register_span_goal_role": "diagnostic_only"' in prompt
    assert '"minimum_pitch": 21' in prompt
    assert '"placement_boundary_is_usage_goal": false' in prompt
    diagnostic = json.loads(
        (run_dir / "outputs/register-diagnostic.json").read_text(encoding="utf-8")
    )
    assert diagnostic["register_enforcement"]["promotion_blocker"] is True
    assert diagnostic["normal_allowed_pitch_range"]["minimum_pitch"] == normal[
        "minimum_pitch"
    ]
    assert diagnostic["placement_allowed_pitch_range"]["minimum_pitch"] == 21


def test_prepared_texture_budget_drift_stops_before_external_call(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    plan, skeleton = _skeleton(48)
    payload = assemble_whole_score_melodies(
        "texture-budget-drift-test",
        plan,
        skeleton,
        tuple(_melody((66, 69, 66, 62)) for _ in skeleton.materials),
    )
    target = next(
        item
        for item in json.loads(
            (run_dir / "inputs/prompt-target.json").read_text(encoding="utf-8")
        )["semantic_targets"]["score_spec"]
        if item["id"] == "attack_texture"
    )
    budget = allocate_texture_budget(
        plan,
        skeleton,
        payload,
        target,
        minimum_attack_group_count=314,
    )
    budget["materials"][0]["required_texture_event_count"] += 1
    (run_dir / "inputs/prepared-texture-budget.json").write_text(
        json.dumps(budget, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runner = _FakeRunner([])

    with pytest.raises(WholeScoreLiveRunError, match="prepared budget"):
        execute_texture_stage(PROJECT_ROOT, run_dir, runner, skeleton, payload)

    assert runner.call_number == 0


def test_unreachable_onset_capacity_budget_stops_before_external_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    plan, skeleton = _skeleton(48)
    payload = assemble_whole_score_melodies(
        "onset-capacity-budget-test",
        plan,
        skeleton,
        tuple(_melody((66, 69, 66, 62)) for _ in skeleton.materials),
    )
    monkeypatch.setattr(
        staged_run,
        "texture_budget_onset_capacity_reachability",
        lambda *args, **kwargs: {
            "schema_version": 1,
            "reachable": False,
            "reason": "attack_size_buckets_unassignable",
            "position_count": 48,
            "required_attack_group_count": 1,
            "matched_position_count": 0,
        },
    )
    runner = _FakeRunner([])

    with pytest.raises(WholeScoreLiveRunError, match="onset capacities"):
        execute_texture_stage(PROJECT_ROOT, run_dir, runner, skeleton, payload)

    assert runner.call_number == 0
    diagnostic = json.loads(
        (
            run_dir / "outputs/texture-onset-capacity-reachability.json"
        ).read_text(encoding="utf-8")
    )
    assert diagnostic["reachable"] is False


def test_partial_texture_batch_rejects_onset_capacity_before_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, skeleton = _skeleton(48)
    payload = assemble_whole_score_melodies(
        "onset-capacity-partial-test",
        plan,
        skeleton,
        tuple(_melody((66, 69, 66, 62)) for _ in skeleton.materials),
    )
    first_draft = _budget_textures(plan, skeleton, payload)[0]
    valid_event = replace(first_draft.events[0], register_zone="low")
    repeated = TextureDraft((valid_event,) * 20)
    budget = {
        "materials": [
            {
                "material_id": skeleton.materials[0].material_id,
                "maximum_texture_event_count": 100,
            }
        ],
        "score_spec_target": {"maximum_group_size": 8},
    }
    monkeypatch.setattr(
        staged_run,
        "_validate_texture_draft_budget",
        lambda *args, **kwargs: None,
    )

    def unexpected_placement(*args, **kwargs):
        raise AssertionError("placement must not run after onset capacity failure")

    monkeypatch.setattr(staged_run, "assemble_texture", unexpected_placement)

    with pytest.raises(WholeScoreLiveRunError, match="onset capacity"):
        staged_run._validate_partial_texture_batches(
            plan,
            skeleton,
            payload,
            budget,
            {skeleton.materials[0].material_id: repeated},
            (43, 93),
        )


def test_partial_texture_batch_rejects_event_feasibility_before_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, skeleton = _skeleton(48)
    payload = assemble_whole_score_melodies(
        "event-feasibility-partial-test",
        plan,
        skeleton,
        tuple(_melody((66, 69, 66, 62)) for _ in skeleton.materials),
    )
    draft = TextureDraft(
        (
            TextureEventDraft(0, 0, 1, "root", "high"),
            TextureEventDraft(0, 0, 1, "fifth", "high"),
        )
    )
    budget = {
        "materials": [
            {
                "material_id": skeleton.materials[0].material_id,
                "maximum_texture_event_count": 100,
            }
        ],
        "score_spec_target": {"maximum_group_size": 8},
    }
    monkeypatch.setattr(
        staged_run,
        "_validate_texture_draft_budget",
        lambda *args, **kwargs: None,
    )

    def unexpected_placement(*args, **kwargs):
        raise AssertionError("placement must not run after event feasibility failure")

    monkeypatch.setattr(staged_run, "assemble_texture", unexpected_placement)

    with pytest.raises(WholeScoreLiveRunError, match="event feasibility"):
        staged_run._validate_partial_texture_batches(
            plan,
            skeleton,
            payload,
            budget,
            {skeleton.materials[0].material_id: draft},
            (43, 93),
        )


def test_partial_texture_batch_can_share_direct_score_event_ids() -> None:
    plan, skeleton = _skeleton(48)
    payload = assemble_whole_score_melodies(
        "shared-event-id-test",
        plan,
        skeleton,
        tuple(_melody((66, 69, 66, 62)) for _ in skeleton.materials),
    )
    budget = allocate_texture_budget(
        plan,
        skeleton,
        payload,
        _attack_texture_target(),
        minimum_attack_group_count=314,
    )
    drafts = _budget_textures(plan, skeleton, payload)
    first_material = skeleton.materials[0]
    first_draft = drafts[0]
    drafts_by_material = {first_material.material_id: first_draft}

    validation = staged_run._validate_partial_texture_batches(
        plan,
        skeleton,
        payload,
        budget,
        drafts_by_material,
        (21, 108),
        score_case_id="same-case",
    )
    base_score = apply_harmonic_skeleton(plan, payload, skeleton)
    target = base_score.materials[0]
    _, scored = assemble_texture(
        "same-case-001",
        plan,
        base_score,
        target,
        first_material.harmonies,
        payload.materials[0].notes,
        first_draft,
        maximum_event_count=int(budget["materials"][0]["maximum_texture_event_count"]),
        placement_policy="bounded-backtracking-v4",
        allowed_pitch_range=(21, 108),
    )
    direct_score = replace(
        base_score,
        materials=(scored, *base_score.materials[1:]),
    )
    legacy = staged_run._validate_partial_texture_batches(
        plan,
        skeleton,
        payload,
        budget,
        drafts_by_material,
        (21, 108),
    )

    direct_sha256 = hashlib.sha256(
        dump_score_spec(direct_score).encode("utf-8")
    ).hexdigest()
    assert validation["temporary_score_spec_sha256"] == direct_sha256
    assert legacy["placements"][0]["notes"][0]["event_id"].startswith(
        f"pilot-batch-{skeleton.materials[0].material_id}-"
    )


def test_texture_budget_violation_is_saved_as_a_failed_stage(tmp_path: Path) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    _, skeleton = _skeleton(48)
    payload = execute_melody_stages(
        PROJECT_ROOT,
        run_dir,
        _FakeRunner(
            [
                dump_melody_collection(
                    tuple(_melody((66, 69, 66, 62)) for _ in range(6))
                ),
                dump_melody_collection(
                    tuple(_melody((66, 69, 66, 62)) for _ in range(2))
                ),
            ]
        ),
        skeleton,
    )

    with pytest.raises(WholeScoreLiveRunError, match="texture budget"):
        execute_texture_stage(
            PROJECT_ROOT,
            run_dir,
            _FakeRunner(
                [dump_texture_collection((*((_texture(),) * 7), _ending_texture()))]
            ),
            skeleton,
            payload,
        )

    failure = json.loads(
        (run_dir / "failures/texture-collection.json").read_text(encoding="utf-8")
    )
    assert failure["schema_version"] == 1
    assert "placement" not in failure
    assert failure["step_id"] == "texture-collection"
    assert "texture budget" in failure["detail"]
    step = json.loads(
        (run_dir / "steps/texture-collection.json").read_text(encoding="utf-8")
    )
    assert step["status"] == "failed"


def _performance_drafts() -> tuple[PerformanceOccurrenceDraftV0, ...]:
    profiles = (
        ("savor", "shape", "aligned"),
        ("flow", "steady", "aligned"),
        ("build", "build", "aligned"),
        ("flow", "shape", "aligned"),
        ("savor", "shape", "aligned"),
        ("build", "build", "aligned"),
        ("build", "build", "aligned"),
        ("flow", "steady", "aligned"),
        ("flow", "shape", "aligned"),
        ("flow", "steady", "aligned"),
        ("release", "release", "aligned"),
    )
    return tuple(
        PerformanceOccurrenceDraftV0(
            timing,
            "subtle",
            dynamics,
            "legato",
            coordination,
            "harmony_legato",
        )
        for timing, dynamics, coordination in profiles
    )


def test_performance_stage_renders_180_second_staged_musicxml_and_smf(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    _allow_any_key_release_target(run_dir)
    prompt_target = _install_long_form_intent(run_dir)
    _, skeleton = _skeleton(48)
    first_pass = (
        _melody((66, 69, 66, 62)),
        _melody((69, 66, 62, 66)),
        _melody((62, 66, 69, 66)),
        _melody((69, 66, 62, 66)),
        _melody((66, 69, 62, 66)),
        _melody((66, 69, 66, 62)),
    )
    transitions = (
        _melody((66, 69, 66, 62)),
        _melody((66, 69, 66, 62)),
    )
    melody_runner = _FakeRunner(
        [dump_melody_collection(first_pass), dump_melody_collection(transitions)]
    )
    payload = execute_melody_stages(
        PROJECT_ROOT,
        run_dir,
        melody_runner,
        skeleton,
    )
    texture_runner = _FakeRunner(
        [dump_texture_collection(_budget_textures(_plan(), skeleton, payload))]
    )
    texture = execute_texture_stage(
        PROJECT_ROOT,
        run_dir,
        texture_runner,
        skeleton,
        payload,
    )
    runner = _FakeRunner([dump_performance_collection(_performance_drafts())])

    result = execute_performance_stage(
        PROJECT_ROOT,
        run_dir,
        runner,
        texture.score,
    )

    assert runner.call_number == 1
    assert result.rendered.duration_ms == 180_000
    assert result.smf_path == run_dir / "staged/final.mid"
    assert result.smf_path.is_file()
    assert (run_dir / "staged/final.musicxml").is_file()
    assert (run_dir / "outputs/performance-spec.dsl").is_file()
    calibration = json.loads(
        (run_dir / "outputs/key-release-calibration.json").read_text(encoding="utf-8")
    )
    assert calibration["status"] == "pass"
    assert calibration["selected_percent"] == result.performance.key_release_percent
    assert 40 <= result.performance.key_release_percent <= 100
    assert '"harmony_count"' in runner.calls[0][1]
    assert '"is_release": true' in runner.calls[0][1]
    assert '"coordination_preservation"' in runner.calls[0][1]
    assert '"key_held_texture"' in runner.calls[0][1]
    assert '"performance_texture.polyphony_duration"' not in runner.calls[0][1]
    assert [call[0] for call in melody_runner.calls] == [
        "melody-collection-main",
        "melody-collection-transition",
    ]
    for _, prompt, hashes in melody_runner.calls:
        assert "creative_climax_melody_collection" in prompt
        assert hashes["creative_targets"] == creative_targets_sha256(
            prompt_target, "melody_collection"
        )
    assert "creative_climax_texture_collection" in texture_runner.calls[0][1]
    assert texture_runner.calls[0][2]["creative_targets"] == creative_targets_sha256(
        prompt_target, "texture_collection"
    )
    assert "creative_climax_performance_spec" in runner.calls[0][1]
    assert runner.calls[0][2]["creative_targets"] == creative_targets_sha256(
        prompt_target, "performance_spec"
    )

    evaluation = evaluate_staged_candidate(PROJECT_ROOT, run_dir)

    assert evaluation["status"] == "completed_unfit"
    validation = json.loads(
        (run_dir / "outputs/rendered-output-validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["status"] == "passed"
    assert validation["musicxml"]["status"] == "passed"
    assert validation["smf"]["status"] == "passed"
    validation_step = json.loads(
        (run_dir / "steps/rendered-output-validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation_step["status"] == "completed"
    assert validation_step["outputs"]["diagnostic_sha256"] == hashlib.sha256(
        (run_dir / "outputs/rendered-output-validation.json").read_bytes()
    ).hexdigest()
    assert evaluation["attack_frequency"]["within_candidate_range"] is False
    assert not (run_dir / "outputs/final.mid").exists()
    assert evaluation["descriptor_count"] == 17
    assert evaluation["copy_risk"]["exact"] is False
    assert set(evaluation["semantic_texture"]) == {
        "attack_texture",
        "key_held_texture",
    }
    assert evaluation["semantic_texture"]["attack_texture"][
        "group_tolerance_ms"
    ] == 30
    assert evaluation["semantic_texture"]["attack_texture"][
        "notes_per_attack_within_neighborhood"
    ] is True
    assert evaluation["semantic_texture"]["attack_texture"][
        "maximum_group_size_limit"
    ] == 8
    assert evaluation["texture_fit"] is True
    assert evaluation["texture_alignment"]["material_count"] == 8
    assert evaluation["texture_alignment"]["occurrence_count"] == 11
    assert all(
        0 <= item["performed_alignment_30ms"] <= 1
        for item in evaluation["texture_alignment"]["occurrences"]
    )


def test_rendered_texture_budget_violation_is_saved_as_a_failed_stage(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    _allow_any_key_release_target(run_dir)
    plan, skeleton = _skeleton(48)
    melody_drafts, _ = normalize_melody_ending(
        plan,
        skeleton,
        tuple(_melody((66, 69, 66, 62)) for _ in skeleton.materials),
    )
    payload = assemble_whole_score_melodies(
        "rendered-budget-test",
        plan,
        skeleton,
        melody_drafts,
    )
    texture = execute_texture_stage(
        PROJECT_ROOT,
        run_dir,
        _FakeRunner(
            [dump_texture_collection(_budget_textures(plan, skeleton, payload))]
        ),
        skeleton,
        payload,
    )
    rolled = tuple(
        replace(item, coordination_profile="rolled")
        for item in _performance_drafts()
    )

    with pytest.raises(WholeScoreLiveRunError, match="texture budget"):
        execute_performance_stage(
            PROJECT_ROOT,
            run_dir,
            _FakeRunner([dump_performance_collection(rolled)]),
            texture.score,
        )

    failure = json.loads(
        (run_dir / "failures/performance-collection.json").read_text(encoding="utf-8")
    )
    assert failure["step_id"] == "performance-collection"
    assert "texture budget" in failure["detail"]


def _all_stage_responses() -> list[str]:
    harmonies = [_harmonic_material(48) for _ in range(8)]
    harmonies[-1] = WholeHarmonicMaterialDraftV0(
        48,
        HarmonicDraft((HarmonicEventDraft(0, 48, 2, "major"),)),
    )
    main = (
        _melody((66, 69, 66, 62)),
        _melody((69, 66, 62, 66)),
        _melody((62, 66, 69, 66)),
        _melody((69, 66, 62, 66)),
        _melody((66, 69, 62, 66)),
        _melody((66, 69, 66, 62)),
    )
    transitions = (
        _melody((66, 69, 66, 62)),
        _melody((66, 69, 66, 62)),
    )
    plan, skeleton = _skeleton(48)
    ordered = (
        main[0],
        main[1],
        main[2],
        transitions[0],
        main[3],
        transitions[1],
        main[4],
        main[5],
    )
    normalized, _ = normalize_melody_ending(plan, skeleton, ordered)
    payload = assemble_whole_score_melodies(
        "response-fixture",
        plan,
        skeleton,
        normalized,
    )
    return [
        dump_harmonic_collection(harmonies),
        dump_melody_collection(main),
        dump_melody_collection(transitions),
        dump_texture_collection(_budget_textures(plan, skeleton, payload)),
        dump_performance_collection(_performance_drafts()),
    ]


def test_prepared_run_uses_exactly_five_calls_without_retry(tmp_path: Path) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    _allow_any_key_release_target(run_dir)
    runner = _FakeRunner(_all_stage_responses())

    result = execute_prepared_whole_score_live_run(PROJECT_ROOT, run_dir, runner)

    assert runner.call_number == 5
    assert result["status"] == "completed_unfit"
    assert result["confirmed_external_call_count"] == 5
    assert not runner.responses


def test_prepared_run_stops_before_evaluation_when_rendered_output_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    _allow_any_key_release_target(run_dir)

    def reject_musicxml(*_args: object) -> dict[str, object]:
        return {
            "status": "failed",
            "path": "staged/final.musicxml",
            "sha256": "0" * 64,
            "checks": [
                {
                    "check_id": "musicxml.notes",
                    "status": "failed",
                    "expected": "score",
                    "actual": "corrupt",
                }
            ],
        }

    monkeypatch.setattr(staged_run, "validate_musicxml_round_trip", reject_musicxml)
    runner = _FakeRunner(_all_stage_responses())

    result = execute_prepared_whole_score_live_run(PROJECT_ROOT, run_dir, runner)

    assert result["status"] == "failed"
    assert result["passes"] is False
    assert result["promoted"] is False
    assert result["error"]["detail"] == "rendered output round-trip validation failed"
    diagnostic = json.loads(
        (run_dir / "outputs/rendered-output-validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["status"] == "failed"
    step = json.loads(
        (run_dir / "steps/rendered-output-validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert step["status"] == "failed"
    assert not (run_dir / "outputs/candidate-evaluation.json").exists()
    assert not (run_dir / "outputs/final.mid").exists()
    assert not (run_dir / "outputs/final.musicxml").exists()


def test_prepared_run_stops_after_invalid_harmonic_ending(tmp_path: Path) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    harmonies = [_harmonic_material(48) for _ in range(8)]
    harmonies[-1] = WholeHarmonicMaterialDraftV0(
        48,
        HarmonicDraft((HarmonicEventDraft(0, 48, 3, "major"),)),
    )
    runner = _FakeRunner([dump_harmonic_collection(harmonies), "must-not-be-used"])

    result = execute_prepared_whole_score_live_run(PROJECT_ROOT, run_dir, runner)

    assert runner.call_number == 1
    assert result["status"] == "failed"
    assert result["error"]["type"] == "WholeScoreLiveRunError"
    assert runner.responses == ["must-not-be-used"]


def test_prepared_run_stops_before_performance_when_score_quality_fails(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "live"
    prepare_whole_score_live_run(PROJECT_ROOT, run_dir)
    responses = _all_stage_responses()
    valid_textures = list(parse_texture_collection(responses[3]))
    ending = valid_textures[-1]
    final_onset = max(event.at_units for event in ending.events)
    valid_textures[-1] = TextureDraft(
        tuple(
            replace(event, duration_units=1)
            if event.at_units == final_onset and event.degree in {"root", "fifth"}
            else event
            for event in ending.events
        )
    )
    responses[3] = dump_texture_collection(valid_textures)
    responses[4] = "must-not-be-used"
    runner = _FakeRunner(responses)

    result = execute_prepared_whole_score_live_run(PROJECT_ROOT, run_dir, runner)

    assert runner.call_number == 4
    assert result["status"] == "failed"
    assert "shared ending chord" in result["error"]["detail"]
    assert runner.responses == ["must-not-be-used"]


def test_cli_entrypoint_runs_only_after_all_helpers_are_defined() -> None:
    source = (
        PROJECT_ROOT
        / "src/llm_musical_composer/whole_score_staged_generation_run.py"
    ).read_text(encoding="utf-8")

    assert source.rfind('if __name__ == "__main__"') > source.rfind(
        "def validate_early_ending"
    )


def test_harmony_prompt_lists_exact_supported_quality_vocabulary() -> None:
    prompt = (
        PROJECT_ROOT / "prompts/pipeline-whole-harmonic-collection.md"
    ).read_text(encoding="utf-8")

    for quality in HARMONY_INTERVALS:
        assert f"`{quality}`" in prompt
    for unsupported in ("augmented", "dominant7", "major7", "minor7"):
        assert f"`{unsupported}`" not in prompt


def test_melody_prompt_matches_supported_foreground_dissonance_contract() -> None:
    prompt = (
        PROJECT_ROOT / "prompts/pipeline-whole-melody-collection.md"
    ).read_text(encoding="utf-8")

    assert "導入と離脱の両方" in prompt
    assert "同じ和声区間内" in prompt
    assert "経過音または隣接音" in prompt
    assert "掛留" not in prompt


def test_texture_prompt_states_the_non_monophonic_material_gate() -> None:
    prompt = (
        PROJECT_ROOT / "prompts/pipeline-whole-texture-collection.md"
    ).read_text(encoding="utf-8")

    assert "すべての素材" in prompt
    assert "同じ発音位置に2イベント以上" in prompt
    assert "同じ和声区間内に3つ以上の異なる発音位置" in prompt
    assert "2種類以上のdegree" in prompt
    assert "実験上の配置許容境界" in prompt
    assert "使用を目指す音域" in prompt


def test_prompts_state_release_and_pedal_quality_contracts() -> None:
    melody = (
        PROJECT_ROOT / "prompts/pipeline-whole-melody-collection.md"
    ).read_text(encoding="utf-8")
    texture = (
        PROJECT_ROOT / "prompts/pipeline-whole-texture-collection.md"
    ).read_text(encoding="utf-8")
    performance = (
        PROJECT_ROOT / "prompts/pipeline-whole-performance-collection.md"
    ).read_text(encoding="utf-8")

    assert "required_final_attack_units" in melody
    assert "required_final_duration_units" in melody
    assert "required_final_attack_units" in texture
    assert "required_final_duration_units" in texture
    assert "required_final_pitch_classes" in texture
    assert "共同終止打鍵後まで持ち越しません" in texture
    assert "複数の和声" in performance
    assert "phrase_legato" in performance
    assert "releaseでは`rolled`を使いません" in performance
