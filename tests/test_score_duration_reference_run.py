from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from llm_musical_composer.pipeline_dsl import (
    parse_performance_spec,
    parse_piece_plan,
    parse_score_spec,
)
from llm_musical_composer.score_duration_reference_run import (
    ScoreDurationReferenceError,
    apply_score_duration_patch,
    build_duration_target_view,
    count_cross_voice_pitch_overlaps,
    material_context,
    parse_score_duration_patch,
    rendered_duration_distribution,
    run_reference_pair,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = (
    PROJECT_ROOT
    / ".appendix/key-release-calibration-v1/reference-v7-a2-joint-texture-v4"
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


@pytest.fixture(scope="module")
def fixed_inputs():
    plan = parse_piece_plan(
        (BASE_ROOT / "inputs/piece-plan.dsl").read_text(encoding="utf-8")
    )
    score = parse_score_spec(
        (BASE_ROOT / "outputs/score-spec.dsl").read_text(encoding="utf-8")
    )
    performance = parse_performance_spec(
        (BASE_ROOT / "outputs/performance-spec.dsl").read_text(encoding="utf-8")
    )
    return plan, score, performance


def _duration_target(path: Path) -> dict:
    target = json.loads(path.read_text(encoding="utf-8"))
    return next(
        item
        for item in target["stage_targets"]["rendered_surface"]
        if item["id"] == "rhythm_time.duration_ratio"
    )


def test_duration_target_view_has_meanings_and_baseline_residuals() -> None:
    target = _duration_target(
        PROJECT_ROOT
        / ".appendix/whole-score-staged-generation-live-v4/"
        "reference-v7-a2-joint-texture-v4/inputs/prompt-target.json"
    )
    baseline = [0.1, 0.2, 0.3, 0.15, 0.15, 0.1]

    view = build_duration_target_view(target, baseline)

    assert [item["id"] for item in view["bins"]] == [
        "very_short",
        "short",
        "near_one_short",
        "near_one_long",
        "long",
        "very_long",
    ]
    assert view["bins"][0]["meaning"] == "全曲中央値発音間隔の0.5倍以下"
    assert view["bins"][-1]["meaning"] == "全曲中央値発音間隔の2倍より長い"
    assert view["baseline_at_key_release_100"] == baseline
    assert view["residual_from_center"][0] == pytest.approx(
        baseline[0] - target["neighborhood_center"][0]
    )


def test_material_context_expands_reused_materials_and_marks_release(fixed_inputs) -> None:
    plan, score, _ = fixed_inputs

    context = material_context(plan, score)

    assert len(context) == 8
    assert sum(item["occurrence_count"] for item in context) == 11
    assert [item["occurrence_count"] for item in context[:2]] == [3, 2]
    assert sum(bool(item["release_fixed"]) for item in context) == 1


def test_patch_changes_only_known_non_release_duration(fixed_inputs) -> None:
    plan, score, _ = fixed_inputs
    context = material_context(plan, score)
    release_ids = {
        item["material_id"] for item in context if item["release_fixed"]
    }
    material = next(
        item
        for item in score.materials
        if item.material_id not in release_ids
        and any(note.duration_units > 1 for note in item.notes)
    )
    note = next(item for item in material.notes if item.duration_units > 1)
    new_duration = note.duration_units - 1
    source = (
        "score_duration_patch(edits=[duration_edit(event_id="
        f"{note.event_id!r}, duration_units={new_duration})])"
    )

    patch = parse_score_duration_patch(source)
    changed = apply_score_duration_patch(plan, score, patch)

    assert changed != score
    assert [item.event_id for item in changed.materials[0].notes] == [
        item.event_id for item in score.materials[0].notes
    ]
    changed_note = next(
        item
        for item in next(
            item for item in changed.materials if item.material_id == material.material_id
        ).notes
        if item.event_id == note.event_id
    )
    assert changed_note == replace(note, duration_units=new_duration)


@pytest.mark.parametrize(
    "source, message",
    [
        ("score_duration_patch(edits=[])", "at least one"),
        (
            "score_duration_patch(edits=["
            "duration_edit(event_id='missing', duration_units=1)])",
            "unknown",
        ),
        (
            "score_duration_patch(edits=["
            "duration_edit(event_id='x', duration_units=1),"
            "duration_edit(event_id='x', duration_units=2)])",
            "duplicated",
        ),
        (
            "score_duration_patch(edits=["
            "duration_edit(event_id='x', duration_units=1.5)])",
            "integer",
        ),
    ],
)
def test_patch_rejects_invalid_edits(fixed_inputs, source: str, message: str) -> None:
    plan, score, _ = fixed_inputs
    with pytest.raises(ScoreDurationReferenceError, match=message):
        apply_score_duration_patch(plan, score, parse_score_duration_patch(source))


def test_patch_rejects_release_material_and_material_overflow(fixed_inputs) -> None:
    plan, score, _ = fixed_inputs
    context = material_context(plan, score)
    release_id = next(item["material_id"] for item in context if item["release_fixed"])
    release = next(item for item in score.materials if item.material_id == release_id)
    release_source = (
        "score_duration_patch(edits=[duration_edit(event_id="
        f"{release.notes[0].event_id!r}, duration_units=1)])"
    )
    with pytest.raises(ScoreDurationReferenceError, match="release"):
        apply_score_duration_patch(plan, score, parse_score_duration_patch(release_source))

    material = next(item for item in score.materials if item.material_id != release_id)
    note = material.notes[0]
    overflow_source = (
        "score_duration_patch(edits=[duration_edit(event_id="
        f"{note.event_id!r}, duration_units={material.length_units + 1})])"
    )
    with pytest.raises(ScoreDurationReferenceError, match="material"):
        apply_score_duration_patch(plan, score, parse_score_duration_patch(overflow_source))


def test_cross_voice_same_pitch_overlap_is_counted(fixed_inputs) -> None:
    _, score, _ = fixed_inputs
    material = score.materials[0]
    upper = next(item for item in material.notes if item.voice == "upper")
    lower_index = next(
        index for index, item in enumerate(material.notes) if item.voice == "lower"
    )
    lower = material.notes[lower_index]
    overlap = replace(
        lower,
        at_units=upper.at_units,
        duration_units=upper.duration_units,
        pitch=upper.pitch,
    )
    changed_material = replace(
        material,
        notes=(
            *material.notes[:lower_index],
            overlap,
            *material.notes[lower_index + 1 :],
        ),
    )
    changed = replace(score, materials=(changed_material, *score.materials[1:]))

    assert count_cross_voice_pitch_overlaps(score) == 0
    assert count_cross_voice_pitch_overlaps(changed) >= 1


def test_duration_measurement_uses_key_release_100(fixed_inputs) -> None:
    plan, score, performance = fixed_inputs

    saved = rendered_duration_distribution(plan, score, performance)
    at_100 = rendered_duration_distribution(
        plan, score, replace(performance, key_release_percent=100)
    )

    assert saved == at_100
    assert sum(saved) == pytest.approx(1.0)
    assert len(saved) == 6


def test_pair_run_calls_each_direction_once_and_keeps_targets_anonymous(
    tmp_path: Path, fixed_inputs
) -> None:
    plan, score, _ = fixed_inputs
    release_ids = {
        item["material_id"]
        for item in material_context(plan, score)
        if item["release_fixed"]
    }
    material = next(
        item
        for item in score.materials
        if item.material_id not in release_ids
        and any(note.duration_units > 1 for note in item.notes)
    )
    note = next(item for item in material.notes if item.duration_units > 1)
    response = (
        "score_duration_patch(edits=[duration_edit(event_id="
        f"{note.event_id!r}, duration_units={note.duration_units - 1})])"
    )
    runner = _FakeRunner([response, response])

    result = run_reference_pair(PROJECT_ROOT, tmp_path / "run", runner)

    assert result["confirmed_external_call_count"] == 2
    assert [item[0] for item in runner.calls] == ["candidate-a", "candidate-b"]
    for _, prompt, _ in runner.calls:
        assert "TasteOfFall" not in prompt
        assert "UnderTheColdSky" not in prompt
        assert not any(value in prompt for value in EXPECTED_TARGET_HASHES)


def test_invalid_first_response_is_not_promoted_and_second_direction_still_runs(
    tmp_path: Path, fixed_inputs
) -> None:
    plan, score, _ = fixed_inputs
    release_ids = {
        item["material_id"]
        for item in material_context(plan, score)
        if item["release_fixed"]
    }
    note = next(
        note
        for material in score.materials
        if material.material_id not in release_ids
        for note in material.notes
        if note.duration_units > 1
    )
    valid = (
        "score_duration_patch(edits=[duration_edit(event_id="
        f"{note.event_id!r}, duration_units={note.duration_units - 1})])"
    )
    runner = _FakeRunner(
        [
            "score_duration_patch(edits=["
            "duration_edit(event_id='unknown', duration_units=1)])",
            valid,
        ]
    )
    run_root = tmp_path / "run"

    result = run_reference_pair(PROJECT_ROOT, run_root, runner)

    assert [item["status"] for item in result["candidates"]] == ["failed", "completed"]
    assert runner.call_number == 2
    assert not (run_root / "candidate-a/outputs/score-spec.dsl").exists()


EXPECTED_TARGET_HASHES = {
    "b7f3575b0edfff704ccfca30b6b8420a316c1d58f48bc166d822821e0c8d8ec4",
    "4549b609273b1a2d3045a8208b4d1118aaab2230eecdd9431d58989910b912b1",
}
