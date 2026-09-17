import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from scoim.performance_ir import PerformanceSpec, SectionPerformance
from scoim.projection_ledger import ProjectionLedgerEntry, validate_projection_ledger
from scoim.score_ir import ScoreDirection, ScoreHarmony, ScoreNote
from scoim.score_projection import (
    PlanChoice,
    build_piece_plan,
    build_score_spec,
)
from scoim.score_rendering import (
    check_rendered_performance_smf,
    check_score_musicxml,
    render_score_performance,
    write_rendered_performance_smf,
    write_score_musicxml,
)
from scoim.script_0_3_validation import check_script_0_3_document
from scoim.script_0_4_validation import check_script_0_4_document
from scoim.validation import CheckResult

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "scoim" / "score-unit-layer-vertical"


@pytest.mark.parametrize("fixture_name", ["basic", "multivoice-offset"])
def test_fixed_fixture_preserves_every_source_through_musicxml_and_smf(
    fixture_name: str, tmp_path: Path
) -> None:
    script = _read_json(FIXTURE_ROOT / f"{fixture_name}-validated-script.json")
    values = _read_json(FIXTURE_ROOT / f"{fixture_name}-validated-stage-values.json")
    _require_phase_boundaries(values)

    script_result = _check_script(script)
    assert script_result.valid, script_result.issues

    phase3 = cast(dict[str, Any], values["phase3"])
    choice = PlanChoice(**phase3["plan_choice"])
    plan, plan_ledger = build_piece_plan(script, choice)

    phase4 = cast(dict[str, Any], values["phase4"])
    phase5 = cast(dict[str, Any], values["phase5"])
    phase6 = cast(dict[str, Any], values["phase6"])
    notes_by_placement = _read_notes_by_placement(
        phase4["notes_by_material_placement"],
        phase5["notes_by_material_placement"],
        phase6["ending_notes_by_material_placement"],
    )
    score, score_ledger = build_score_spec(
        script,
        plan,
        score_id=f"{fixture_name}-score",
        divisions=phase3["divisions"],
        length_units_by_score_unit=phase3["length_units_by_score_unit"],
        harmonies_by_score_unit={
            unit_id: tuple(ScoreHarmony(**item) for item in items)
            for unit_id, items in phase3["harmonies_by_score_unit"].items()
        },
        directions_by_score_unit={
            unit_id: tuple(ScoreDirection(**item) for item in items)
            for unit_id, items in phase6["directions_by_score_unit"].items()
        },
        notes_by_material_placement=notes_by_placement,
        cumulative_projection_ledger=_upstream_score_ledger(plan, notes_by_placement),
    )

    phase7 = cast(dict[str, Any], values["phase7"])
    performance = PerformanceSpec(
        performance_id=phase7["performance_id"],
        target_duration_ms=phase7["target_duration_ms"],
        default_velocity=phase7["default_velocity"],
        timing_budget_id=phase7["timing_budget_id"],
        section_performances=tuple(
            SectionPerformance(**item) for item in phase7["section_performances"]
        ),
    )
    rendered = render_score_performance(script, plan, score, performance)

    ledger = (*plan_ledger, *score_ledger)
    validate_projection_ledger(ledger)
    assert all(entry.status == "passed" and entry.evidence for entry in ledger)

    script_body = cast(dict[str, Any], script["script"])
    leaf_section_ids = {
        section_id
        for section_id in script_body["sections"]
        if not any(
            section["parent_section_id"] == section_id
            for section in script_body["sections"].values()
        )
    }
    assert {unit.source_section_id for unit in score.score_units} == leaf_section_ids
    assert {
        layer.source_material_placement_id
        for unit in score.score_units
        for layer in unit.score_unit_layers
    } == set(script_body["material_placements"])
    assert {note.source_score_note_id for note in rendered.notes} == {
        note.score_note_id
        for unit in score.score_units
        for layer in unit.score_unit_layers
        for note in layer.notes
    }

    musicxml_path = write_score_musicxml(script, plan, score, tmp_path / f"{fixture_name}.musicxml")
    smf_path = write_rendered_performance_smf(rendered, tmp_path / f"{fixture_name}.mid")
    assert check_score_musicxml(script, plan, score, musicxml_path)["status"] == "passed"
    assert check_rendered_performance_smf(rendered, smf_path)["status"] == "passed"

    if fixture_name == "multivoice-offset":
        texture_unit = next(
            unit for unit in score.score_units if unit.source_section_id == "texture"
        )
        wide_layer = next(
            layer
            for layer in texture_unit.score_unit_layers
            if layer.source_material_placement_id == "wide-theme-first"
        )
        support_layer = next(
            layer
            for layer in texture_unit.score_unit_layers
            if layer.source_material_placement_id == "offset-support-first"
        )
        assert {note.voice for note in wide_layer.notes} == {"upper", "lower"}
        assert _note_span(wide_layer.notes) == (0, 48)
        assert _note_span(support_layer.notes) == (6, 36)
        assert len(plan.nodes) == 3
        assert len(score.score_units) == 2


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _check_script(script: Mapping[str, object]) -> CheckResult:
    if script.get("schema_version") == "0.4.0":
        return check_script_0_4_document(script)
    return check_script_0_3_document(script)


def _require_phase_boundaries(values: Mapping[str, object]) -> None:
    assert set(values) == {"phase3", "phase4", "phase5", "phase6", "phase7"}
    expected_keys = {
        "phase3": {
            "plan_choice",
            "divisions",
            "length_units_by_score_unit",
            "harmonies_by_score_unit",
        },
        "phase4": {"notes_by_material_placement"},
        "phase5": {"notes_by_material_placement"},
        "phase6": {
            "ending_notes_by_material_placement",
            "directions_by_score_unit",
        },
        "phase7": {
            "performance_id",
            "target_duration_ms",
            "default_velocity",
            "timing_budget_id",
            "section_performances",
        },
    }
    for phase, keys in expected_keys.items():
        assert isinstance(values[phase], dict)
        assert set(cast(dict[str, object], values[phase])) == keys


def _read_notes_by_placement(
    *phase_note_maps: Mapping[str, list[dict[str, Any]]],
) -> dict[str, tuple[ScoreNote, ...]]:
    result: dict[str, tuple[ScoreNote, ...]] = {}
    for note_map in phase_note_maps:
        assert result.keys().isdisjoint(note_map)
        result.update(
            {
                placement_id: tuple(ScoreNote(**note) for note in notes)
                for placement_id, notes in note_map.items()
            }
        )
    return result


def _note_span(notes: tuple[ScoreNote, ...]) -> tuple[int, int]:
    return min(note.at_units for note in notes), max(
        note.at_units + note.duration_units for note in notes
    )


def _upstream_score_ledger(plan, notes_by_placement):
    targets = {
        "plan_node": tuple(node.section_id for node in plan.nodes),
        "score_unit": tuple(
            node.score_unit_id for node in plan.nodes if node.score_unit_id is not None
        ),
        "score_unit_layer": tuple(
            f"score-unit-layer-{placement_id}" for placement_id in notes_by_placement
        ),
        "score_note": tuple(
            note.score_note_id for notes in notes_by_placement.values() for note in notes
        ),
    }
    return tuple(
        ProjectionLedgerEntry(
            "fixture",
            target_id,
            target_kind,
            target_id,
            "upstream",
            "direct_id_equality",
            "passed",
            f"{target_kind}_id={target_id}",
        )
        for target_kind, target_ids in targets.items()
        for target_id in target_ids
    )
