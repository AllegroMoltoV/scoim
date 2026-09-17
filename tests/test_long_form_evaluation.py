from __future__ import annotations

from dataclasses import replace

from llm_musical_composer.composition_ir import Material, Note
from llm_musical_composer.long_form_evaluation import evaluate_long_form_structure
from llm_musical_composer.music_dsl import THREE_MINUTE_POLICY, parse_composition
from tests.test_music_dsl import LONG_PARTS_SOURCE, PHRASE_PARTS_SOURCE


def _material(material_id: str, count: int, velocity: int, chord_size: int) -> Material:
    notes = tuple(
        Note(
            event_id=f"{material_id.lower()}{index}",
            at_ms=(index // chord_size) * 2_000,
            duration_ms=800 + (index % 3) * 200,
            pitch=48 + index,
            velocity=velocity + (index % 2),
        )
        for index in range(count)
    )
    return Material(material_id, 44_000, notes)


def _composition(*, climax_count: int = 12):
    plan = parse_composition(
        LONG_PARTS_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    return replace(
        plan,
        materials=(
            _material("A", 3, 50, 1),
            _material("B", 6, 68, 2),
            _material("C", climax_count, 92, 4),
        ),
    )


def test_long_form_evaluation_observes_climax_decline_and_exact_return() -> None:
    report = evaluate_long_form_structure(_composition())

    assert report["status"] == "pass"
    assert report["climax"]["part_id"] == "P3"
    assert len(report["climax"]["supporting_metrics"]) >= 2
    assert len(report["post_climax_decline"]["supporting_metrics"]) >= 2
    assert report["return"]["reused_material_ids"] == ["A"]
    assert len(report["parts"]) == 4
    assert len(report["boundaries"]) == 3


def test_declared_climax_without_actual_activity_is_reported_as_failure() -> None:
    report = evaluate_long_form_structure(_composition(climax_count=1))

    assert report["status"] == "fail"
    assert "climax" in " ".join(report["issues"])


def test_phrase_evaluation_reports_declared_variation_and_return_separately() -> None:
    plan = parse_composition(
        PHRASE_PARTS_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    composition = replace(
        plan,
        materials=(
            _material("A", 3, 50, 1),
            replace(_material("B", 6, 68, 2), derived_from="A"),
            _material("C", 12, 92, 3),
        ),
    )

    report = evaluate_long_form_structure(composition)

    assert [phrase["phrase_id"] for phrase in report["phrases"]] == [
        "S1",
        "V1",
        "C1",
        "R1",
    ]
    assert report["phrases"][1]["derived_from"] == "S1"
    assert report["phrases"][1]["variation_kind"] is None
    assert report["phrases"][1]["derived_material_ids"] == ["B"]
    assert report["phrases"][3]["reused_source_material_ids"] == ["A"]
