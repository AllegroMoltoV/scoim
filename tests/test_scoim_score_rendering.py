from dataclasses import replace

import pytest

from scoim.performance_ir import (
    PerformanceIrValidationError,
    PerformanceSpec,
    SectionPerformance,
    validate_rendered_performance,
)
from scoim.score_ir import (
    PiecePlan,
    PlanNode,
    ScoreHarmony,
    ScoreNote,
    ScoreSpec,
    ScoreUnit,
    ScoreUnitLayer,
)
from scoim.score_rendering import (
    ScoreRenderingError,
    _legacy_score_boundary,
    check_rendered_performance_smf,
    check_score_musicxml,
    render_score_performance,
    write_rendered_performance_smf,
    write_score_musicxml,
)
from scoim.validation import IssueCode


def _script() -> dict[str, object]:
    return {
        "script": {
            "root_section_id": "whole",
            "sections": {
                "whole": {"parent_section_id": None, "order": 0, "role": "whole"},
                "statement": {
                    "parent_section_id": "whole",
                    "order": 0,
                    "role": "statement",
                },
            },
            "material_placements": {
                "theme-first": {
                    "section_id": "statement",
                    "material_id": "theme",
                    "role": "foreground",
                },
                "support-first": {
                    "section_id": "statement",
                    "material_id": "support",
                    "role": "accompaniment",
                },
            },
            "script_element_variation_relations": {},
        }
    }


def _score_inputs() -> tuple[PiecePlan, ScoreSpec, PerformanceSpec]:
    plan = PiecePlan(
        "plan-001",
        "小さな曲",
        0,
        "major",
        "whole",
        (
            PlanNode("whole", None, 0),
            PlanNode("statement", "whole", 0, 1, "score-unit-statement"),
        ),
    )
    score = ScoreSpec(
        "score-001",
        12,
        (
            ScoreUnit(
                "score-unit-statement",
                "statement",
                48,
                (ScoreHarmony("harmony-001", 0, 48, 0, "major"),),
                (),
                (
                    ScoreUnitLayer(
                        "layer-theme",
                        "theme-first",
                        (
                            ScoreNote("note-upper", 0, 24, 72, "upper"),
                            ScoreNote("note-lower", 12, 12, 48, "lower"),
                        ),
                    ),
                    ScoreUnitLayer(
                        "layer-support",
                        "support-first",
                        (ScoreNote("note-support", 0, 48, 43, "lower"),),
                    ),
                ),
            ),
        ),
    )
    performance = PerformanceSpec(
        "performance-001",
        180_000,
        64,
        "subtle-v1",
        (
            SectionPerformance(
                "whole",
                timing_profile="neutral",
                timing_amount="subtle",
                dynamics_profile="steady",
                articulation_profile="score",
                coordination_profile="score",
                pedal_profile="none",
            ),
        ),
    )
    return plan, score, performance


def _placement_variation_inputs() -> tuple[
    dict[str, object], PiecePlan, ScoreSpec, PerformanceSpec
]:
    script = _script()
    script["script"]["sections"]["return"] = {
        "parent_section_id": "whole",
        "order": 1,
        "role": "return",
    }
    script["script"]["material_placements"]["theme-return"] = {
        "section_id": "return",
        "material_id": "theme",
        "role": "foreground",
    }
    script["script"]["script_element_variation_relations"]["theme-varied"] = {
        "source": {"type": "material_placement", "id": "theme-first"},
        "target": {"type": "material_placement", "id": "theme-return"},
    }
    plan, score, performance = _score_inputs()
    plan = replace(
        plan,
        nodes=(
            *plan.nodes,
            PlanNode("return", "whole", 1, 1, "score-unit-return"),
        ),
    )
    return_unit = ScoreUnit(
        "score-unit-return",
        "return",
        48,
        (ScoreHarmony("harmony-return", 0, 48, 0, "major"),),
        (),
        (
            ScoreUnitLayer(
                "layer-theme-return",
                "theme-return",
                (ScoreNote("note-return", 0, 48, 72, "upper"),),
            ),
        ),
    )
    return script, plan, replace(score, score_units=(*score.score_units, return_unit)), performance


def test_render_score_performance_preserves_every_score_note_source() -> None:
    plan, score, performance = _score_inputs()

    rendered = render_score_performance(_script(), plan, score, performance)
    validate_rendered_performance(plan, score, performance, rendered)

    assert {note.source_score_note_id for note in rendered.notes} == {
        "note-upper",
        "note-lower",
        "note-support",
    }
    assert len({note.performed_note_id for note in rendered.notes}) == 3
    assert rendered.duration_ms == 180_000
    assert len(rendered.piece_plan_sha256) == 64
    assert len(rendered.score_spec_sha256) == 64
    assert len(rendered.performance_spec_sha256) == 64


def test_rendered_performance_rejects_a_duplicate_score_note_source() -> None:
    plan, score, performance = _score_inputs()
    rendered = render_score_performance(_script(), plan, score, performance)
    duplicate_source = replace(
        rendered.notes[1],
        source_score_note_id=rendered.notes[0].source_score_note_id,
    )
    invalid_rendered = replace(
        rendered,
        notes=(rendered.notes[0], duplicate_source, *rendered.notes[2:]),
    )

    with pytest.raises(PerformanceIrValidationError, match="one to one"):
        validate_rendered_performance(plan, score, performance, invalid_rendered)


def test_rendered_performance_rejects_an_unknown_pedal_score_unit() -> None:
    plan, score, performance = _score_inputs()
    rendered = render_score_performance(_script(), plan, score, performance)
    invalid_rendered = replace(
        rendered,
        pedals=(replace(rendered.pedals[0], source_score_unit_id="missing"),),
    )

    with pytest.raises(PerformanceIrValidationError, match="unknown score unit"):
        validate_rendered_performance(plan, score, performance, invalid_rendered)


def test_rendered_performance_rejects_missing_section_intervals() -> None:
    plan, score, performance = _score_inputs()
    rendered = render_score_performance(_script(), plan, score, performance)

    with pytest.raises(PerformanceIrValidationError, match="cover the plan"):
        validate_rendered_performance(
            plan,
            score,
            performance,
            replace(rendered, section_intervals=()),
        )


def test_rendered_performance_rejects_a_lineage_hash_mismatch() -> None:
    plan, score, performance = _score_inputs()
    rendered = render_score_performance(_script(), plan, score, performance)

    with pytest.raises(PerformanceIrValidationError, match="lineage hash"):
        validate_rendered_performance(
            plan,
            score,
            performance,
            replace(rendered, score_spec_sha256="0" * 64),
        )


def test_score_and_performance_can_be_written_to_musicxml_and_smf(tmp_path) -> None:
    plan, score, performance = _score_inputs()
    rendered = render_score_performance(_script(), plan, score, performance)

    musicxml_path = write_score_musicxml(_script(), plan, score, tmp_path / "score.musicxml")
    smf_path = write_rendered_performance_smf(rendered, tmp_path / "performance.mid")

    assert musicxml_path.is_file()
    assert smf_path.is_file()
    assert check_score_musicxml(_script(), plan, score, musicxml_path)["status"] == "passed"
    assert check_rendered_performance_smf(rendered, smf_path)["status"] == "passed"


def test_rendering_preserves_a_placement_variation_through_a_return_section() -> None:
    script, plan, score, performance = _placement_variation_inputs()

    rendered = render_score_performance(script, plan, score, performance)

    assert {note.source_score_note_id for note in rendered.notes} == {
        "note-upper",
        "note-lower",
        "note-support",
        "note-return",
    }


def test_rendering_does_not_apply_legacy_section_role_rules_to_v2_sections() -> None:
    script, plan, score, performance = _placement_variation_inputs()
    script["script"]["sections"]["return"]["role"] = "homecoming"
    script["script"]["script_element_variation_relations"] = {}

    rendered = render_score_performance(script, plan, score, performance)

    assert {note.source_score_note_id for note in rendered.notes} == {
        "note-upper",
        "note-lower",
        "note-support",
        "note-return",
    }


def test_rendering_does_not_infer_foreground_from_score_note_voices() -> None:
    plan, score, performance = _score_inputs()

    _, legacy_score = _legacy_score_boundary(_script(), plan, score)

    assert all(material.foreground_voice is None for material in legacy_score.materials)
    render_score_performance(_script(), plan, score, performance)


def test_rendering_rejects_an_unsupported_velocity_policy_without_reinterpreting_it() -> None:
    plan, score, performance = _score_inputs()
    unsupported = replace(
        performance,
        velocity_policy_id="foreground-accompaniment-harmony-shape-v1",
    )

    with pytest.raises(ScoreRenderingError) as raised:
        render_score_performance(_script(), plan, score, unsupported)

    assert raised.value.issue.code is IssueCode.UNREPRESENTABLE
    assert raised.value.issue.path == "/performance/velocity_policy_id"


def test_rendering_assigns_harmony_pedal_events_to_the_active_score_unit() -> None:
    plan, score, performance = _score_inputs()
    root_performance = replace(
        performance.section_performances[0],
        pedal_profile="harmony_legato",
    )

    rendered = render_score_performance(
        _script(),
        plan,
        score,
        replace(performance, section_performances=(root_performance,)),
    )

    assert rendered.pedals[0].at_ms == 0
    assert all(pedal.source_score_unit_id == "score-unit-statement" for pedal in rendered.pedals)


def test_rendering_keeps_the_starting_score_unit_for_a_pedal_up_at_a_boundary() -> None:
    script, plan, score, performance = _placement_variation_inputs()
    root_performance = replace(
        performance.section_performances[0],
        pedal_profile="harmony_legato",
    )

    rendered = render_score_performance(
        script,
        plan,
        score,
        replace(performance, section_performances=(root_performance,)),
    )

    first_pedal_up = next(
        pedal for pedal in rendered.pedals if pedal.performed_pedal_id == "pedal-0-0-up"
    )
    assert first_pedal_up.source_score_unit_id == "score-unit-statement"


def test_rendering_rejects_script_sections_that_do_not_match_the_plan() -> None:
    plan, score, performance = _score_inputs()
    script = _script()
    script["script"]["sections"]["extra"] = {
        "parent_section_id": "whole",
        "order": 1,
        "role": "statement",
    }

    with pytest.raises(ScoreRenderingError, match="sections do not match"):
        render_score_performance(script, plan, score, performance)


def test_rendering_rejects_script_placements_that_do_not_match_the_score() -> None:
    plan, score, performance = _score_inputs()
    script = _script()
    del script["script"]["material_placements"]["support-first"]

    with pytest.raises(ScoreRenderingError, match="layers do not match"):
        render_score_performance(script, plan, score, performance)
