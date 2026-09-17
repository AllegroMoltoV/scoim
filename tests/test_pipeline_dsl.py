from __future__ import annotations

import pytest

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    PipelineValidationError,
    PlanNode,
    validate_pipeline,
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
from llm_musical_composer.pipeline_prompts import (
    TEMPLATES,
    PipelinePromptError,
    build_stage_prompt,
)

PLAN_SOURCE = """piece_plan(
  plan_id="p",
  title="段階別DSL",
  tonal_center=9,
  mode="minor",
  root_node_id="root",
  ending_intent="tonic",
  nodes=[
    plan_node(node_id="root", parent_id=None, order=0, role="whole"),
    plan_node(
      node_id="leaf",
      parent_id="root",
      order=0,
      role="statement",
      harmonic_focus=9,
      duration_weight=1,
      score_material_id="theme",
    ),
  ],
)"""

SCORE_SOURCE = """score_spec(
  score_id="s",
  divisions=4,
  materials=[
    score_material(
      material_id="theme",
      length_units=8,
      notes=[
        score_note(
          event_id="u",
          at_units=0,
          duration_units=8,
          pitch=64,
          voice="upper",
          articulations=["tenuto"],
        ),
        score_note(
          event_id="l",
          at_units=0,
          duration_units=8,
          pitch=48,
          voice="lower",
        ),
      ],
      directions=[
        score_direction(
          direction_id="d",
          at_units=0,
          kind="dynamic",
          value="mf",
        ),
      ],
    ),
  ],
)"""

PERFORMANCE_SOURCE = """performance_spec(
  performance_id="x",
  target_duration_ms=12000,
  default_velocity=64,
  timing_budget_id="subtle-v1",
  node_performances=[
    node_performance(
      node_id="root",
      timing_profile="savor",
      timing_amount="moderate",
      dynamics_profile="shape",
      articulation_profile="legato",
      pedal_profile="phrase_legato",
    ),
  ],
)"""

HARMONIC_SCORE_SOURCE = """score_spec(
  score_id="harmonic",
  divisions=4,
  materials=[
    score_material(
      material_id="theme",
      length_units=8,
      harmonies=[
        score_harmony(
          harmony_id="am",
          at_units=0,
          duration_units=4,
          root_pitch_class=9,
          quality="minor",
        ),
        score_harmony(
          harmony_id="c",
          at_units=4,
          duration_units=4,
          root_pitch_class=0,
          quality="major",
        ),
      ],
      foreground_voice="upper",
      notes=[
        score_note(event_id="u", at_units=0, duration_units=8, pitch=69, voice="upper"),
        score_note(event_id="l1", at_units=0, duration_units=4, pitch=45, voice="lower"),
        score_note(event_id="l2", at_units=4, duration_units=4, pitch=48, voice="lower"),
      ],
    ),
  ],
)"""


def test_stage_dsls_round_trip_and_validate_together() -> None:
    plan = parse_piece_plan(PLAN_SOURCE)
    score = parse_score_spec(SCORE_SOURCE)
    performance = parse_performance_spec(PERFORMANCE_SOURCE)

    validate_pipeline(plan, score, performance)

    assert parse_piece_plan(dump_piece_plan(plan)) == plan
    assert parse_score_spec(dump_score_spec(score)) == score
    assert parse_performance_spec(dump_performance_spec(performance)) == performance
    assert dump_piece_plan(parse_piece_plan(dump_piece_plan(plan))) == dump_piece_plan(plan)


def test_performance_key_release_percent_is_backward_compatible() -> None:
    legacy = parse_performance_spec(PERFORMANCE_SOURCE)

    assert legacy.key_release_percent == 100
    assert legacy.velocity_policy_id == "legacy-unison-v1"
    assert "key_release_percent" not in dump_performance_spec(legacy)
    assert "velocity_policy_id" not in dump_performance_spec(legacy)

    source = PERFORMANCE_SOURCE.replace(
        '  node_performances=[',
        '  key_release_percent=48,\n  node_performances=[',
    )
    calibrated = parse_performance_spec(source)

    assert calibrated.key_release_percent == 48
    assert "key_release_percent=48" in dump_performance_spec(calibrated)
    assert parse_performance_spec(dump_performance_spec(calibrated)) == calibrated


def test_performance_velocity_policy_round_trips_without_changing_legacy_dump() -> None:
    legacy = parse_performance_spec(PERFORMANCE_SOURCE)
    legacy_dump = dump_performance_spec(legacy)
    source = PERFORMANCE_SOURCE.replace(
        '  node_performances=[',
        '  velocity_policy_id="foreground-accompaniment-harmony-shape-v1",\n'
        '  node_performances=[',
    )

    shaped = parse_performance_spec(source)

    assert dump_performance_spec(parse_performance_spec(legacy_dump)) == legacy_dump
    assert shaped.velocity_policy_id == "foreground-accompaniment-harmony-shape-v1"
    assert "velocity_policy_id='foreground-accompaniment-harmony-shape-v1'" in (
        dump_performance_spec(shaped)
    )
    assert parse_performance_spec(dump_performance_spec(shaped)) == shaped


def test_performance_velocity_policy_rejects_unknown_id() -> None:
    plan = parse_piece_plan(PLAN_SOURCE)
    score = parse_score_spec(SCORE_SOURCE)
    source = PERFORMANCE_SOURCE.replace(
        '  node_performances=[',
        '  velocity_policy_id="unknown-v1",\n  node_performances=[',
    )

    with pytest.raises(PipelineValidationError, match="velocity policy"):
        validate_pipeline(plan, score, parse_performance_spec(source))


@pytest.mark.parametrize("percent", [39, 101])
def test_performance_key_release_percent_rejects_out_of_range(percent: int) -> None:
    plan = parse_piece_plan(PLAN_SOURCE)
    score = parse_score_spec(SCORE_SOURCE)
    source = PERFORMANCE_SOURCE.replace(
        '  node_performances=[',
        f'  key_release_percent={percent},\n  node_performances=[',
    )

    with pytest.raises(PipelineValidationError, match="key release"):
        validate_pipeline(plan, score, parse_performance_spec(source))


def test_harmonic_score_dsl_round_trips_deterministically() -> None:
    score = parse_score_spec(HARMONIC_SCORE_SOURCE)

    dumped = dump_score_spec(score)

    assert parse_score_spec(dumped) == score
    assert dump_score_spec(parse_score_spec(dumped)) == dumped
    assert score.materials[0].foreground_voice == "upper"
    assert [item.quality for item in score.materials[0].harmonies] == ["minor", "major"]


def test_major_seventh_harmony_round_trips_deterministically() -> None:
    source = HARMONIC_SCORE_SOURCE.replace(
        'quality="major"',
        'quality="major-seventh"',
    )

    score = parse_score_spec(source)
    dumped = dump_score_spec(score)

    assert score.materials[0].harmonies[1].quality == "major-seventh"
    assert parse_score_spec(dumped) == score
    assert dump_score_spec(parse_score_spec(dumped)) == dumped


def test_arbitrary_depth_calibration_fixture_round_trips_across_all_stages() -> None:
    plan = PiecePlan(
        plan_id="multiscale",
        title="多尺度校正",
        tonal_center=9,
        mode="minor",
        root_node_id="root",
        ending_intent="tonic",
        nodes=(
            PlanNode("root", None, 0, "whole"),
            PlanNode("a1", "root", 0, "opening"),
            PlanNode("a1-inner", "a1", 0, "statement"),
            PlanNode(
                "a1-leaf",
                "a1-inner",
                0,
                "statement",
                harmonic_focus=9,
                duration_weight=1,
                score_material_id="theme",
            ),
            PlanNode("b", "root", 1, "contrast", contrasts_with="a1"),
            PlanNode("b-inner", "b", 0, "contrast"),
            PlanNode(
                "b-leaf",
                "b-inner",
                0,
                "statement",
                harmonic_focus=0,
                duration_weight=1,
                score_material_id="theme",
            ),
            PlanNode("a2", "root", 2, "return", derived_from="a1"),
            PlanNode(
                "a2-inner",
                "a2",
                0,
                "return",
                derived_from="a1-inner",
            ),
            PlanNode(
                "a2-leaf",
                "a2-inner",
                0,
                "return",
                derived_from="a1-leaf",
                harmonic_focus=9,
                duration_weight=1,
                score_material_id="theme",
            ),
        ),
    )
    score = parse_score_spec(SCORE_SOURCE)
    performance = PerformanceSpec(
        performance_id="multiscale-performance",
        target_duration_ms=180_000,
        default_velocity=64,
        timing_budget_id="subtle-v1",
        node_performances=(
            NodePerformance(
                "a1",
                timing_profile="savor",
                timing_amount="moderate",
                coordination_profile="rolled",
                pedal_profile="phrase_legato",
            ),
            NodePerformance(
                "b",
                timing_profile="build",
                timing_amount="subtle",
                pedal_profile="clear",
            ),
            NodePerformance(
                "a2",
                timing_profile="flow",
                timing_amount="subtle",
                coordination_profile="aligned",
                pedal_profile="phrase_legato",
            ),
        ),
    )

    reparsed_plan = parse_piece_plan(dump_piece_plan(plan))
    reparsed_score = parse_score_spec(dump_score_spec(score))
    reparsed_performance = parse_performance_spec(dump_performance_spec(performance))

    validate_pipeline(reparsed_plan, reparsed_score, reparsed_performance)
    assert reparsed_plan == plan
    assert reparsed_score == score
    assert reparsed_performance == performance


@pytest.mark.parametrize(
    ("parser", "source", "message"),
    (
        (parse_piece_plan, '__import__("os")', "unknown DSL function"),
        (
            parse_piece_plan,
            PLAN_SOURCE.replace('role="statement"', 'role="statement", pitch=60'),
            "unknown argument",
        ),
        (
            parse_score_spec,
            SCORE_SOURCE.replace('voice="upper"', 'voice="upper", velocity=64'),
            "unknown argument",
        ),
        (
            parse_performance_spec,
            PERFORMANCE_SOURCE.replace('node_id="root"', 'node_id="root", pitch=60'),
            "unknown argument",
        ),
        (
            parse_score_spec,
            SCORE_SOURCE.replace("notes=[", "notes=[note for note in []] or ["),
            "list literal",
        ),
    ),
)
def test_stage_dsls_reject_code_and_cross_stage_fields(parser, source, message) -> None:
    with pytest.raises(PipelineDslError, match=message):
        parser(source)


@pytest.mark.parametrize(
    ("stage", "values", "required_boundary"),
    (
        (
            "piece_plan",
            {
                "REQUEST_JSON": "{}",
                "REFERENCE_TARGET_JSON": "{}",
                "CALIBRATION_CONTRACT": "A1 -> B -> A2",
            },
            "禁止フィールド: 音符、pitch",
        ),
        (
            "score_spec",
            {
                "PIECE_PLAN_DSL": PLAN_SOURCE,
                "REFERENCE_TARGET_JSON": "{}",
                "SOURCE_MATERIALS_DSL": "[]",
                "IDENTITY_CUES": "pitch_contour",
            },
            "禁止フィールド: at_ms",
        ),
        (
            "performance_spec",
            {
                "PIECE_PLAN_DSL": PLAN_SOURCE,
                "REFERENCE_TARGET_JSON": "{}",
                "SCORE_SUMMARY": "one material",
                "PROFILE_CATALOG": "savor, flow",
            },
            "禁止フィールド: pitch",
        ),
    ),
)
def test_stage_prompts_keep_inputs_and_forbidden_outputs_separate(
    stage, values, required_boundary
) -> None:
    prompt = build_stage_prompt(stage, values)

    assert required_boundary in prompt
    assert all(value in prompt for value in values.values())
    assert "{{" not in prompt


def test_stage_prompt_rejects_missing_or_cross_stage_inputs() -> None:
    with pytest.raises(PipelinePromptError, match="mismatch"):
        build_stage_prompt(
            "piece_plan",
            {
                "REQUEST_JSON": "{}",
                "REFERENCE_TARGET_JSON": "{}",
                "SCORE_SUMMARY": "cross-stage",
            },
        )


def test_stage_prompts_define_the_exact_dsl_signatures_needed_by_an_isolated_model() -> None:
    piece = build_stage_prompt(
        "piece_plan",
        {"REQUEST_JSON": "{}", "REFERENCE_TARGET_JSON": "{}", "CALIBRATION_CONTRACT": "{}"},
    )
    score = build_stage_prompt(
        "score_spec",
        {
            "PIECE_PLAN_DSL": PLAN_SOURCE,
            "REFERENCE_TARGET_JSON": "{}",
            "SOURCE_MATERIALS_DSL": "[]",
            "IDENTITY_CUES": "{}",
        },
    )
    performance = build_stage_prompt(
        "performance_spec",
        {
            "PIECE_PLAN_DSL": PLAN_SOURCE,
            "REFERENCE_TARGET_JSON": "{}",
            "SCORE_SUMMARY": "{}",
            "PROFILE_CATALOG": "{}",
        },
    )

    assert "piece_plan(plan_id=..., title=..." in piece
    assert "plan_node(node_id=..., parent_id=..." in piece
    assert "score_spec(score_id=..., divisions=..." in score
    assert "score_material(material_id=..., length_units=..." in score
    assert "score_note(event_id=..., at_units=..." in score
    assert "performance_spec(performance_id=..., target_duration_ms=..." in performance
    assert "node_performance(node_id=..." in performance


@pytest.mark.parametrize(
    ("source", "message"),
    (
        ("piece_plan(", "invalid syntax"),
        ("factory.piece_plan()", "direct calls"),
        ("piece_plan('p')", "positional arguments"),
        ("piece_plan(**{})", "keyword expansion"),
        (
            'piece_plan(plan_id="a", plan_id="b")',
            "duplicate argument",
        ),
        ('piece_plan(plan_id="a")', "missing argument"),
        (PLAN_SOURCE.replace("tonal_center=9", 'tonal_center="9"'), "literal int"),
    ),
)
def test_piece_plan_dsl_reports_structural_syntax_failures(source, message) -> None:
    with pytest.raises(PipelineDslError, match=message):
        parse_piece_plan(source)


def test_stage_prompt_rejects_template_drift_and_unresolved_insertions(
    tmp_path, monkeypatch
) -> None:
    broken = tmp_path / "broken.md"
    broken.write_text("{{REQUEST_JSON}}", encoding="utf-8")
    monkeypatch.setitem(TEMPLATES, "piece_plan", broken)
    values = {
        "REQUEST_JSON": "{}",
        "REFERENCE_TARGET_JSON": "{}",
        "CALIBRATION_CONTRACT": "known",
    }
    with pytest.raises(PipelinePromptError, match="placeholders"):
        build_stage_prompt("piece_plan", values)

    valid = tmp_path / "valid.md"
    valid.write_text(
        "{{REQUEST_JSON}} {{REFERENCE_TARGET_JSON}} {{CALIBRATION_CONTRACT}}",
        encoding="utf-8",
    )
    monkeypatch.setitem(TEMPLATES, "piece_plan", valid)
    values["CALIBRATION_CONTRACT"] = "{{UNRESOLVED}}"
    with pytest.raises(PipelinePromptError, match="unresolved"):
        build_stage_prompt("piece_plan", values)
