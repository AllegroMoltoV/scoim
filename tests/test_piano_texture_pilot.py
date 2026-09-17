from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from llm_musical_composer.generic_pipeline_quality import evaluate_generic_score_quality
from llm_musical_composer.performance_pipeline import (
    PiecePlan,
    PlanNode,
    ScoreDirection,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)
from llm_musical_composer.piano_texture_pilot import (
    PianoTextureEvent,
    PianoTextureSpec,
    PianoTextureValidationError,
    build_fixed_material_view,
    combine_piano_texture,
    dump_piano_texture_spec,
    execute_piano_texture_pilot,
    parse_piano_texture_spec,
    prepare_piano_texture_pilot,
    validate_and_convert_piano_texture,
    validate_combination_scope,
)
from llm_musical_composer.pipeline_dsl import dump_piece_plan, dump_score_spec


def _fixture() -> tuple[PiecePlan, ScoreSpec]:
    plan = PiecePlan(
        "texture-plan",
        "伴奏境界パイロット",
        2,
        "major",
        "entire",
        "tonic",
        (
            PlanNode("entire", None, 0, "whole"),
            PlanNode(
                "statement",
                "entire",
                0,
                "statement",
                duration_weight=1,
                score_material_id="material_1",
            ),
            PlanNode(
                "contrast",
                "entire",
                1,
                "contrast",
                contrasts_with="statement",
                duration_weight=1,
                score_material_id="material_2",
            ),
            PlanNode(
                "return",
                "entire",
                2,
                "return",
                derived_from="statement",
                duration_weight=1,
                score_material_id="material_1",
            ),
            PlanNode(
                "ending",
                "entire",
                3,
                "release",
                duration_weight=1,
                score_material_id="material_4",
            ),
        ),
    )
    material_1 = ScoreMaterial(
        "material_1",
        16,
        (
            ScoreNote("m1_l1", 0, 8, 38, "lower"),
            ScoreNote("m1_l2", 0, 8, 45, "lower"),
            ScoreNote("m1_u1", 0, 8, 66, "upper"),
            ScoreNote("m1_l3", 8, 8, 45, "lower"),
            ScoreNote("m1_u2", 8, 8, 69, "upper"),
        ),
        harmonies=(ScoreHarmony("m1_h1", 0, 16, 2, "major"),),
        foreground_voice="upper",
    )
    material_2 = ScoreMaterial(
        "material_2",
        16,
        (
            ScoreNote("m2_l1", 0, 8, 43, "lower"),
            ScoreNote("m2_l2", 0, 8, 50, "lower"),
            ScoreNote("m2_u1", 0, 4, 71, "upper"),
            ScoreNote("m2_u2", 4, 4, 74, "upper"),
            ScoreNote("m2_u3", 8, 8, 71, "upper"),
        ),
        harmonies=(ScoreHarmony("m2_h1", 0, 16, 7, "major"),),
        foreground_voice="upper",
    )
    material_4 = ScoreMaterial(
        "material_4",
        32,
        (
            ScoreNote("m4_u01", 0, 4, 78, "upper", articulations=("tenuto",)),
            ScoreNote("m4_u02", 4, 4, 81, "upper"),
            ScoreNote("m4_u03", 8, 4, 78, "upper", articulations=("tenuto",)),
            ScoreNote("m4_u04", 12, 4, 74, "upper"),
            ScoreNote("m4_u05", 16, 4, 69, "upper", articulations=("tenuto",)),
            ScoreNote("m4_u06", 20, 4, 66, "upper"),
            ScoreNote("m4_u07", 24, 8, 74, "upper", articulations=("tenuto",)),
            ScoreNote("m4_l01", 0, 8, 38, "lower", articulations=("tenuto",)),
            ScoreNote("m4_l02", 0, 8, 45, "lower", articulations=("tenuto",)),
            ScoreNote("m4_l03", 8, 8, 42, "lower", articulations=("tenuto",)),
            ScoreNote("m4_l04", 8, 8, 50, "lower", articulations=("tenuto",)),
            ScoreNote("m4_l05", 16, 8, 45, "lower", articulations=("tenuto",)),
            ScoreNote("m4_l06", 16, 8, 52, "lower", articulations=("tenuto",)),
            ScoreNote("m4_l07", 24, 8, 38, "lower", articulations=("tenuto",)),
            ScoreNote("m4_l08", 24, 8, 45, "lower", articulations=("tenuto",)),
        ),
        directions=(ScoreDirection("m4_d1", 0, "dynamic", "mp"),),
        harmonies=(ScoreHarmony("m4_h1", 0, 32, 2, "major"),),
        foreground_voice="upper",
    )
    return plan, ScoreSpec("texture-score", 8, (material_1, material_2, material_4))


def _valid_texture() -> PianoTextureSpec:
    return PianoTextureSpec(
        material_id="material_4",
        events=(
            PianoTextureEvent(
                "pt_01", "m4_h1", "accompaniment", "lower", 0, 8, "root", 2, ("tenuto",)
            ),
            PianoTextureEvent(
                "pt_02", "m4_h1", "accompaniment", "lower", 0, 8, "fifth", 2, ("tenuto",)
            ),
            PianoTextureEvent(
                "pt_03", "m4_h1", "accompaniment", "lower", 8, 8, "third", 2, ("tenuto",)
            ),
            PianoTextureEvent(
                "pt_04", "m4_h1", "accompaniment", "lower", 8, 8, "root", 3, ("tenuto",)
            ),
            PianoTextureEvent(
                "pt_05", "m4_h1", "accompaniment", "lower", 16, 8, "root", 2, ("tenuto",)
            ),
            PianoTextureEvent(
                "pt_06", "m4_h1", "accompaniment", "lower", 16, 8, "fifth", 2, ("tenuto",)
            ),
            PianoTextureEvent(
                "pt_07", "m4_h1", "accompaniment", "lower", 24, 8, "root", 2, ("tenuto",)
            ),
            PianoTextureEvent(
                "pt_08", "m4_h1", "accompaniment", "lower", 24, 8, "fifth", 2, ("tenuto",)
            ),
        ),
    )


def test_piano_texture_dsl_round_trips_deterministically() -> None:
    texture = _valid_texture()

    source = dump_piano_texture_spec(texture)

    assert parse_piano_texture_spec(source) == texture
    assert dump_piano_texture_spec(parse_piano_texture_spec(source)) == source
    assert "pitch=" not in source


@pytest.mark.parametrize(
    "source",
    (
        "piano_texture_spec('material_4', events=[])",
        "piano_texture_spec(material_id='material_4', events=[], pitch=38)",
        "piano_texture_spec(material_id='material_4', events=[score_note(event_id='x')])",
        (
            "piano_texture_spec(material_id='material_4', events=["
            "piano_texture_note(event_id='x', harmony_id='m4_h1', "
            "role='accompaniment', voice='lower', at_units=0.0, "
            "duration_units=8, degree='root', octave=2, articulations=[])])"
        ),
    ),
)
def test_piano_texture_dsl_rejects_out_of_contract_syntax(source: str) -> None:
    with pytest.raises(PianoTextureValidationError):
        parse_piano_texture_spec(source)


def test_fixed_material_view_excludes_old_accompaniment() -> None:
    plan, score = _fixture()

    view = build_fixed_material_view(plan, score, "material_4")
    encoded = json.dumps(view, ensure_ascii=False, sort_keys=True)

    assert view["piece_plan"] == {"tonal_center": 2, "mode": "major"}
    assert view["score"] == {"score_id": "texture-score", "divisions": 8}
    assert len(view["material"]["foreground_notes"]) == 7
    assert "m4_u01" in encoded
    assert "m4_l01" not in encoded
    assert '"pitch": 52' not in encoded


def test_valid_texture_replaces_all_accompaniment_and_passes_existing_quality() -> None:
    plan, score = _fixture()
    texture = _valid_texture()

    notes = validate_and_convert_piano_texture(plan, score, texture)
    combined = combine_piano_texture(plan, score, texture)

    assert len(notes) == 8
    assert all(note.tie is None for note in notes)
    assert {note.pitch for note in notes} == {38, 42, 45, 50}
    validate_combination_scope(score, combined, "material_4")
    target = next(item for item in combined.materials if item.material_id == "material_4")
    assert not any(note.event_id.startswith("m4_l") for note in target.notes)
    result = evaluate_generic_score_quality(plan, combined)
    assert result["passes"] is True
    harmony = next(item for item in result["harmony"] if item["material_id"] == "material_4")
    assert harmony["accompaniment_chord_tone_ratio"] == 1.0


def test_texture_rejects_voice_crossing_even_when_pitch_is_a_chord_tone() -> None:
    plan, score = _fixture()
    events = list(_valid_texture().events)
    events[0] = replace(events[0], degree="fifth", octave=6)

    with pytest.raises(PianoTextureValidationError, match="voice crossing"):
        validate_and_convert_piano_texture(
            plan, score, replace(_valid_texture(), events=tuple(events))
        )


def test_texture_rejects_one_note_degenerate_accompaniment() -> None:
    plan, score = _fixture()

    with pytest.raises(PianoTextureValidationError, match="exactly 8 events"):
        validate_and_convert_piano_texture(
            plan, score, replace(_valid_texture(), events=(_valid_texture().events[-1],))
        )


def test_texture_rejects_missing_attack_cell() -> None:
    plan, score = _fixture()
    events = list(_valid_texture().events)
    events[1] = replace(events[1], at_units=4)
    events[4] = replace(events[4], at_units=8)
    events[5] = replace(events[5], at_units=8)

    with pytest.raises(PianoTextureValidationError, match="each divisions-sized cell"):
        validate_and_convert_piano_texture(
            plan, score, replace(_valid_texture(), events=tuple(events))
        )


def test_texture_rejects_seventh_for_a_triad() -> None:
    plan, score = _fixture()
    events = list(_valid_texture().events)
    events[0] = replace(events[0], degree="seventh")

    with pytest.raises(PianoTextureValidationError, match="degree"):
        validate_and_convert_piano_texture(
            plan, score, replace(_valid_texture(), events=tuple(events))
        )


def test_combination_scope_detects_a_fixed_field_change() -> None:
    _, score = _fixture()
    materials = list(score.materials)
    target = materials[-1]
    materials[-1] = replace(target, length_units=31)

    with pytest.raises(PianoTextureValidationError, match="fixed score content changed"):
        validate_combination_scope(score, replace(score, materials=tuple(materials)), "material_4")


class _FakeRunner:
    def __init__(self, source: str) -> None:
        self.source = source
        self.call_number = 0

    def run(
        self, step_id: str, prompt: str, input_hashes: dict[str, str] | None = None
    ) -> dict[str, object]:
        assert step_id == "generate-piano-texture"
        assert "m4_l06" not in prompt
        assert input_hashes
        self.call_number += 1
        return {"composition_source": self.source, "intent_summary": "伴奏だけを生成"}


def _prepare_fixture(tmp_path: Path) -> Path:
    plan, score = _fixture()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    plan_path = input_dir / "piece-plan.dsl"
    score_path = input_dir / "score-spec-001.dsl"
    plan_path.write_text(dump_piece_plan(plan), encoding="utf-8")
    score_path.write_text(dump_score_spec(score), encoding="utf-8")
    output_dir = tmp_path / "pilot"
    prepare_piano_texture_pilot(
        piece_plan_path=plan_path,
        score_spec_path=score_path,
        output_dir=output_dir,
    )
    return output_dir


def test_prepare_snapshots_inputs_and_hides_old_accompaniment(tmp_path: Path) -> None:
    output_dir = _prepare_fixture(tmp_path)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    fixed = (output_dir / "fixed-material.json").read_text(encoding="utf-8")
    prompt = (output_dir / "prompt.txt").read_text(encoding="utf-8")

    assert manifest["status"] == "prepared"
    assert manifest["passes"] is False
    assert (output_dir / "input-piece-plan.dsl").is_file()
    assert (output_dir / "input-score-spec.dsl").is_file()
    assert "m4_u01" in fixed
    assert "m4_l01" not in fixed
    assert "m4_l06" not in prompt
    assert "piano_texture_spec" in prompt


def test_external_response_is_combined_and_recorded_once(tmp_path: Path) -> None:
    output_dir = _prepare_fixture(tmp_path)
    runner = _FakeRunner(dump_piano_texture_spec(_valid_texture()))

    result = execute_piano_texture_pilot(output_dir=output_dir, runner=runner)

    assert runner.call_number == 1
    assert result["status"] == "completed"
    assert result["passes"] is True
    assert result["source_kind"] == "external"
    assert result["confirmed_external_call_count"] == 1
    assert (output_dir / "response.txt").is_file()
    assert (output_dir / "piano-texture.dsl").is_file()
    assert (output_dir / "combined-score-spec.dsl").is_file()
    assert (output_dir / "score-quality.json").is_file()


def test_invalid_external_response_is_saved_without_retry(tmp_path: Path) -> None:
    output_dir = _prepare_fixture(tmp_path)
    runner = _FakeRunner("not_texture()")

    result = execute_piano_texture_pilot(output_dir=output_dir, runner=runner)

    assert runner.call_number == 1
    assert result["status"] == "failed"
    assert result["passes"] is False
    assert (output_dir / "response.txt").read_text(encoding="utf-8") == "not_texture()"
    assert not (output_dir / "piano-texture.dsl").exists()
    assert not (output_dir / "combined-score-spec.dsl").exists()
