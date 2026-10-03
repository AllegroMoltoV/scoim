from dataclasses import replace

import pytest

from scoim.performance_ir import (
    PerformanceIrValidationError,
    PerformanceSpec,
    PerformedNote,
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
    _merge_simultaneous_key_strikes,
    check_rendered_performance_smf,
    check_score_musicxml,
    render_score_performance,
    write_rendered_performance_smf,
    write_score_musicxml,
)
from scoim.score_timing import QUANTIZED_TIMING
from scoim.validation import IssueCode


@pytest.mark.parametrize("short_interval", ["note", "harmony"])
def test_quantized_rendering_rejects_intervals_lost_at_ms_precision(short_interval) -> None:
    plan, score, performance = _score_inputs()
    script = _script()
    script["script"]["material_placements"].pop("support-first")
    unit = score.score_units[0]
    long_duration = 10**9
    if short_interval == "note":
        harmonies = (replace(unit.harmonies[0], duration_units=long_duration),)
        notes = (ScoreNote("short", 1, 1, 72, "upper"),)
    else:
        harmonies = (
            replace(unit.harmonies[0], duration_units=1),
            ScoreHarmony("remaining", 1, long_duration - 1, 0, "major"),
        )
        notes = (ScoreNote("long", 0, long_duration, 72, "upper"),)
    score = replace(
        score,
        score_units=(
            replace(
                unit,
                length_units=long_duration,
                harmonies=harmonies,
                score_unit_layers=(replace(unit.score_unit_layers[0], notes=notes),),
            ),
        ),
    )
    with pytest.raises(ScoreRenderingError, match=f"score {short_interval} collapses") as error:
        render_score_performance(script, plan, score, performance, timing_contract=QUANTIZED_TIMING)
    assert error.value.issue.code == IssueCode.UNREPRESENTABLE


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


def test_quantized_rendering_preserves_notes_pedals_and_boundaries_when_scaled() -> None:
    plan, score, performance = _score_inputs()
    performance = replace(
        performance,
        timing_budget_id="narrative-v2",
        section_performances=(
            replace(performance.section_performances[0], timing_profile="savor"),
            SectionPerformance(
                "statement",
                timing_profile="build",
                timing_amount="moderate",
                pedal_profile="harmony_legato",
            ),
        ),
    )
    scaled = replace(
        score,
        divisions=score.divisions * 10,
        score_units=tuple(
            replace(
                unit,
                length_units=unit.length_units * 10,
                harmonies=tuple(
                    replace(
                        harmony,
                        at_units=harmony.at_units * 10,
                        duration_units=harmony.duration_units * 10,
                    )
                    for harmony in unit.harmonies
                ),
                score_unit_layers=tuple(
                    replace(
                        layer,
                        notes=tuple(
                            replace(
                                note,
                                at_units=note.at_units * 10,
                                duration_units=note.duration_units * 10,
                            )
                            for note in layer.notes
                        ),
                    )
                    for layer in unit.score_unit_layers
                ),
            )
            for unit in score.score_units
        ),
    )
    original = render_score_performance(
        _script(),
        plan,
        score,
        performance,
        timing_contract=QUANTIZED_TIMING,
    )
    enlarged = render_score_performance(
        _script(),
        plan,
        scaled,
        performance,
        timing_contract=QUANTIZED_TIMING,
    )
    assert enlarged.notes == original.notes
    assert enlarged.pedals == original.pedals
    assert enlarged.section_intervals == original.section_intervals


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


def _with_same_key_unison(score: ScoreSpec) -> ScoreSpec:
    unit = score.score_units[0]
    support_layer = unit.score_unit_layers[1]
    return replace(
        score,
        score_units=(
            replace(
                unit,
                score_unit_layers=(
                    unit.score_unit_layers[0],
                    replace(
                        support_layer,
                        notes=(replace(support_layer.notes[0], pitch=72),),
                    ),
                ),
            ),
        ),
    )


def test_render_score_performance_preserves_every_score_note_source() -> None:
    plan, score, performance = _score_inputs()

    rendered = render_score_performance(_script(), plan, score, performance)
    validate_rendered_performance(plan, score, performance, rendered)

    assert {source_id for note in rendered.notes for source_id in note.source_score_note_ids} == {
        "note-upper",
        "note-lower",
        "note-support",
    }
    assert len({note.performed_note_id for note in rendered.notes}) == 3
    assert all(not hasattr(note, "voice") for note in rendered.notes)
    assert rendered.duration_ms == 180_000
    assert len(rendered.piece_plan_sha256) == 64
    assert len(rendered.score_spec_sha256) == 64
    assert len(rendered.performance_spec_sha256) == 64


def test_rendering_merges_score_simultaneous_same_key_notes_into_one_strike() -> None:
    plan, score, performance = _score_inputs()
    score = _with_same_key_unison(score)

    rendered = render_score_performance(_script(), plan, score, performance)

    merged = [
        note
        for note in rendered.notes
        if set(note.source_score_note_ids) == {"note-upper", "note-support"}
    ]
    assert len(merged) == 1
    assert merged[0].duration_ms == 162_000
    assert len(rendered.notes) == 2


def test_rendering_rolls_same_key_unison_as_one_strike() -> None:
    plan, score, performance = _score_inputs()
    score = _with_same_key_unison(score)
    root_performance = replace(
        performance.section_performances[0],
        coordination_profile="rolled",
    )

    rendered = render_score_performance(
        _script(),
        plan,
        score,
        replace(performance, section_performances=(root_performance,)),
    )

    merged = [
        note
        for note in rendered.notes
        if set(note.source_score_note_ids) == {"note-upper", "note-support"}
    ]
    assert len(merged) == 1
    assert len(rendered.notes) == 2


def test_rendering_rejects_two_velocities_for_one_simultaneous_key() -> None:
    notes = (
        PerformedNote("strike-a", ("score-a",), 100, 500, 60, 64),
        PerformedNote("strike-b", ("score-b",), 100, 700, 60, 72),
    )

    with pytest.raises(ScoreRenderingError) as raised:
        _merge_simultaneous_key_strikes(notes)

    assert raised.value.issue.code is IssueCode.UNREPRESENTABLE


def test_rendering_ends_a_key_before_the_next_strike() -> None:
    plan, score, performance = _score_inputs()
    unit = score.score_units[0]
    support_layer = unit.score_unit_layers[1]
    score = replace(
        score,
        score_units=(
            replace(
                unit,
                score_unit_layers=(
                    unit.score_unit_layers[0],
                    replace(
                        support_layer,
                        notes=(
                            replace(
                                support_layer.notes[0],
                                at_units=12,
                                duration_units=36,
                                pitch=72,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    rendered = render_score_performance(_script(), plan, score, performance)

    same_key = sorted(
        (note for note in rendered.notes if note.pitch == 72),
        key=lambda note: note.at_ms,
    )
    assert len(same_key) == 2
    assert same_key[0].at_ms + same_key[0].duration_ms == same_key[1].at_ms


def test_rendered_performance_rejects_a_duplicate_score_note_source() -> None:
    plan, score, performance = _score_inputs()
    rendered = render_score_performance(_script(), plan, score, performance)
    duplicate_source = replace(
        rendered.notes[1],
        source_score_note_ids=rendered.notes[0].source_score_note_ids,
    )
    invalid_rendered = replace(
        rendered,
        notes=(rendered.notes[0], duplicate_source, *rendered.notes[2:]),
    )

    with pytest.raises(PerformanceIrValidationError, match="exactly once"):
        validate_rendered_performance(plan, score, performance, invalid_rendered)


def test_rendered_performance_rejects_overlapping_strikes_on_one_key() -> None:
    plan, score, performance = _score_inputs()
    rendered = render_score_performance(_script(), plan, score, performance)
    upper = next(note for note in rendered.notes if note.source_score_note_ids == ("note-upper",))
    support = next(
        note for note in rendered.notes if note.source_score_note_ids == ("note-support",)
    )
    invalid_support = replace(
        support,
        at_ms=upper.at_ms + 100,
        duration_ms=upper.duration_ms,
        pitch=upper.pitch,
    )
    invalid_rendered = replace(
        rendered,
        notes=tuple(invalid_support if note is support else note for note in rendered.notes),
    )

    with pytest.raises(PerformanceIrValidationError, match="same piano key"):
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

    assert {source_id for note in rendered.notes for source_id in note.source_score_note_ids} == {
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

    assert {source_id for note in rendered.notes for source_id in note.source_score_note_ids} == {
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

    assert rendered.pedals[0].at_ms == 80
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
        pedal for pedal in rendered.pedals if pedal.performed_pedal_id == "pedal-0-up"
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
