import json
from pathlib import Path

import pytest

from llm_musical_composer import (
    supported_normal_generation_terminal_fallback as fallback_module,
)
from llm_musical_composer.generic_pipeline_quality import (
    evaluate_generic_pipeline_quality,
    evaluate_generic_score_quality,
)
from llm_musical_composer.harmonic_skeleton import parse_harmonic_skeleton
from llm_musical_composer.pipeline_dsl import parse_piece_plan
from llm_musical_composer.texture_budget import (
    measure_rendered_texture_budget,
    measure_score_texture_budget,
)
from llm_musical_composer.whole_score_staged_generation import (
    assemble_whole_score_melodies,
)
from llm_musical_composer.whole_score_staged_generation_dsl import (
    parse_melody_collection,
    parse_texture_collection,
)

PROJECT_ROOT = Path(__file__).parents[1]
V5_ROOT = PROJECT_ROOT / ".appendix/supported-normal-generation-v5-placement-backtracking"
V7_ROOT = PROJECT_ROOT / ".appendix/supported-normal-generation-v7-final-release-only"
V7_RUN = V7_ROOT / "runs/whole-score"


def _fixed_inputs():
    plan = parse_piece_plan(
        (V7_RUN / "inputs/piece-plan.dsl").read_text(encoding="utf-8")
    )
    skeleton = parse_harmonic_skeleton(
        (V7_RUN / "outputs/harmonic-skeleton.dsl").read_text(encoding="utf-8")
    )
    melody_drafts = parse_melody_collection(
        (V7_RUN / "inputs/prepared-melody-collection.dsl").read_text(
            encoding="utf-8"
        )
    )
    melodies = assemble_whole_score_melodies(
        "terminal-fallback-test",
        plan,
        skeleton,
        melody_drafts,
    )
    prefix = parse_texture_collection(
        (V7_ROOT / "inputs/prepared-texture-prefix.dsl").read_text(
            encoding="utf-8"
        )
    )
    final = parse_texture_collection(
        (
            V5_ROOT
            / "runs/whole-score/outputs/texture-collection-batch-003.dsl"
        ).read_text(encoding="utf-8")
    )[-1]
    budget = json.loads(
        (V7_RUN / "inputs/prepared-texture-budget.json").read_text(
            encoding="utf-8"
        )
    )
    prompt_target = json.loads(
        (V7_RUN / "inputs/prompt-target.json").read_text(encoding="utf-8")
    )
    return plan, skeleton, melodies, prefix, final, budget, prompt_target


def test_terminal_texture_repair_preserves_the_exact_score_budget() -> None:
    plan, skeleton, melodies, prefix, final, budget, _ = _fixed_inputs()

    result, drafts, diagnostic = fallback_module.repair_terminal_texture(
        plan,
        skeleton,
        melodies,
        prefix,
        final,
        budget,
    )

    assert len(drafts) == 7
    assert diagnostic["candidate_count"] == 1
    assert diagnostic["selected"] == {
        "source_onset_units": 1,
        "target_onset_units": 12,
        "source_event_index": 3,
        "degree": "fifth",
    }
    assert diagnostic["invariants"] == {
        "event_count_unchanged": True,
        "degree_unchanged": True,
        "register_zone_unchanged": True,
    }
    score_budget = measure_score_texture_budget(plan, result.score, budget)
    assert score_budget["whole_score"]["matches_budget"] is True
    assert score_budget["whole_score"]["attack_group_count"] == 649
    assert score_budget["whole_score"]["note_event_count"] == 1_504
    assert evaluate_generic_score_quality(plan, result.score)["passes"] is True


def test_fixed_performance_preserves_rendered_texture_and_ending() -> None:
    plan, skeleton, melodies, prefix, final, budget, prompt_target = _fixed_inputs()
    result, _, _ = fallback_module.repair_terminal_texture(
        plan,
        skeleton,
        melodies,
        prefix,
        final,
        budget,
    )

    performance, rendered, calibration, diagnostic = (
        fallback_module.build_terminal_performance(
            plan,
            result.score,
            prompt_target,
        )
    )

    assert [
        item.articulation_profile for item in performance.node_performances
    ] == [
        "light",
        "light",
        "legato",
        "legato",
        "light",
        "light",
        "light",
        "legato",
    ]
    assert all(
        item.coordination_profile in {"score", "aligned"}
        for item in performance.node_performances
    )
    assert performance.key_release_percent == 60
    assert calibration["status"] == "pass"
    assert diagnostic["external_call_count"] == 0
    assert evaluate_generic_pipeline_quality(
        plan,
        result.score,
        performance,
        rendered,
    )["passes"] is True
    rendered_budget = measure_rendered_texture_budget(rendered, budget)
    assert rendered_budget["rendered_performance"]["matches_budget"] is True
    assert rendered_budget["rendered_performance"]["frequency_matches_budget"] is True
    assert rendered_budget["rendered_performance"]["texture_shape_matches_budget"] is True
    assert rendered_budget["rendered_performance"]["attack_group_count"] == 649
    assert rendered_budget["rendered_performance"]["note_event_count"] == 1_504


def test_prepare_rejects_tampered_v7_timeout_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original = fallback_module._read_json

    def tampered(path: Path):
        value = original(path)
        if path.name == "run-state.json" and V7_ROOT.name in path.parts:
            value = json.loads(json.dumps(value))
            value["calls"]["saved_response_count"] = 1
        return value

    monkeypatch.setattr(fallback_module, "_read_json", tampered)

    with pytest.raises(
        fallback_module.SupportedNormalGenerationTerminalFallbackError,
        match="v7 timeout provenance",
    ):
        fallback_module.prepare_terminal_fallback(
            PROJECT_ROOT,
            tmp_path / "terminal-fallback",
        )


def test_terminal_fallback_completes_without_external_calls(tmp_path: Path) -> None:
    artifact_root = tmp_path / "terminal-fallback"

    result = fallback_module.run_terminal_fallback(PROJECT_ROOT, artifact_root)

    assert result["status"] == "completed_unfit"
    assert result["passes"] is False
    assert result["promoted"] is False
    assert result["confirmed_external_call_count"] == 0
    assert not (artifact_root / "runs/whole-score/outputs/final.mid").exists()
    assert not (artifact_root / "runs/whole-score/outputs/final.musicxml").exists()
    assert (artifact_root / "runs/whole-score/staged/final.mid").is_file()
    assert result["evaluation"]["candidate_use"] == "feasibility_only"
    assert result["evaluation"]["diagnostic_fit"] is True
