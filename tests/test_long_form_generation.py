from __future__ import annotations

import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest
from test_music_dsl import LONG_PARTS_SOURCE, PHRASE_PARTS_SOURCE, VALID_SOURCE

from llm_musical_composer.composition_ir import Composition, Material, Note, Pedal, Use
from llm_musical_composer.long_form_generation import (
    LongFormGenerationError,
    _batches_for_plan,
    _material_specs,
    _normalize_final_pedal_releases,
    _normalize_forbidden_ending_chords,
    _normalize_note_bounds,
    _normalize_out_of_scale_notes,
    _normalize_same_pitch_retriggers,
    _pitches_form_forbidden_chord,
    _render_prompt,
    _repair_exact_material_copies,
    _repair_failed_variations,
    _restore_registral_pattern,
    _restore_rhythmic_contour,
    _restore_textural_upper_motif,
    _validate_long_form_inner_structure,
    _validate_long_form_phrase_structure,
    _validate_material_variations,
    _validate_natural_long_form_plan,
    _validate_natural_materials,
    _validate_voice_and_variations,
    _warp_rhythmic_material,
    composition_to_source,
    run_staged_generation,
    transpose_composition,
)
from llm_musical_composer.music_dsl import THREE_MINUTE_POLICY, parse_composition


def _material_source(material_id: str) -> str:
    pitch = {"A": 60, "B": 64, "C": 67}[material_id]
    return (
        f'material("{material_id}", duration_ms=44000, notes=['
        f'note("{material_id.lower()}1", at_ms=0, duration_ms=1000, '
        f"pitch={pitch}, velocity=70)], pedals=[])"
    )


def test_transpose_composition_changes_only_pitch_and_tonal_center() -> None:
    original = Composition(
        title="control",
        tonal_center=2,
        mode="minor",
        form=(Use("A", role="opening", energy=2),),
        materials=(
            Material(
                "A",
                5_500,
                (
                    Note("n1", 0, 500, 45, 80, "lower"),
                    Note("n2", 200, 400, 67, 90, "upper"),
                ),
                (Pedal("p1", 0, 127), Pedal("p2", 5_500, 0)),
            ),
        ),
    )

    shifted = transpose_composition(original, 1)

    assert shifted.tonal_center == 3
    assert [note.pitch for note in shifted.materials[0].notes] == [46, 68]
    assert [
        replace(note, pitch=old.pitch)
        for note, old in zip(shifted.materials[0].notes, original.materials[0].notes, strict=True)
    ] == list(original.materials[0].notes)
    assert shifted.form == original.form
    assert shifted.materials[0].pedals == original.materials[0].pedals

    with pytest.raises(LongFormGenerationError, match="outside the piano range"):
        transpose_composition(
            replace(
                original,
                materials=(replace(original.materials[0], notes=(Note("high", 0, 100, 108, 80),)),),
            ),
            1,
        )


def _material_source_for_test(material: Material) -> str:
    notes = ", ".join(
        f'note("{note.event_id}", {note.pitch}, {note.velocity}, '
        f'{note.at_ms}, {note.duration_ms}, voice="{note.voice}")'
        for note in material.notes
    )
    derived = (
        f', derived_from="{material.derived_from}"' if material.derived_from is not None else ""
    )
    return (
        f'material("{material.material_id}", duration_ms={material.duration_ms}, '
        f"notes=[{notes}], pedals=[]{derived})"
    )


class FakeRunner:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    def run(self, step_id: str, prompt: str, input_hashes=None) -> dict[str, object]:
        self.calls.append((step_id, prompt, input_hashes or {}))
        return {"composition_source": next(self.responses), "intent_summary": "test"}


def test_same_pitch_retrigger_clips_only_the_previous_note() -> None:
    material = Material(
        "A",
        2_000,
        (
            Note("a1", 0, 1_000, 60, 80),
            Note("other", 100, 1_200, 64, 70),
            Note("a2", 900, 500, 60, 90),
        ),
        (),
    )

    normalized = _normalize_same_pitch_retriggers((material,))[0]

    assert normalized.notes[0] == Note("a1", 0, 900, 60, 80)
    assert normalized.notes[1:] == material.notes[1:]


def test_note_bounds_drop_non_sounding_events_and_clip_overshoot() -> None:
    material = Material(
        "A",
        2_000,
        (
            Note("negative", -1, 100, 60, 80),
            Note("zero", 100, 0, 62, 80),
            Note("boundary", 2_000, 100, 64, 80),
            Note("clipped", 1_800, 500, 65, 80),
            Note("valid", 500, 300, 67, 80),
        ),
        (),
    )

    normalized = _normalize_note_bounds((material,))[0]

    assert normalized.notes == (
        Note("clipped", 1_800, 200, 65, 80),
        Note("valid", 500, 300, 67, 80),
    )


def test_simultaneous_duplicate_pitch_keeps_the_longest_single_key_event() -> None:
    material = Material(
        "A",
        2_000,
        (
            Note("a1", 0, 1_000, 60, 80),
            Note("a2", 0, 500, 60, 90),
            Note("later", 1_200, 500, 60, 85),
        ),
        (),
    )

    normalized = _normalize_same_pitch_retriggers((material,))[0]

    assert normalized.notes == (material.notes[0], material.notes[2])


def test_plan_and_materials_are_separate_calls_and_assemble_deterministically(
    tmp_path: Path,
) -> None:
    responses = [
        LONG_PARTS_SOURCE,
        f'material_batch("batch-01", materials=[{_material_source("A")}])',
        f'material_batch("batch-02", materials=[{_material_source("B")}])',
        f'material_batch("batch-03", materials=[{_material_source("C")}])',
    ]
    runner = FakeRunner(responses)

    composition = run_staged_generation(
        runner,
        plan_prompt="make a plan",
        material_prompt_template="batch={{batch_id}} specs={{material_specs}}",
        output_dir=tmp_path,
        base_input_hashes={"style_target": "target-hash"},
    )

    assert [call[0] for call in runner.calls] == [
        "plan",
        "material-batch-01",
        "material-batch-02",
        "material-batch-03",
    ]
    assert composition.duration_ms == 180_000
    assert [use.material_id for use in composition.form] == ["A", "B", "C", "A"]
    assert '"part_role": "opening"' in runner.calls[1][1]
    assert '"part_role": "return"' in runner.calls[1][1]
    assert '"use_role": "climax"' in runner.calls[3][1]
    assert runner.calls[0][2] == {"style_target": "target-hash"}
    assert all(call[2]["style_target"] == "target-hash" for call in runner.calls)
    assert "plan" in runner.calls[1][2]
    assert "batch_spec" in runner.calls[1][2]
    assert (
        parse_composition(
            (tmp_path / "final.music.py").read_text(encoding="utf-8"),
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )
        == composition
    )


def test_material_batch_cannot_change_plan_scope(tmp_path: Path) -> None:
    runner = FakeRunner(
        [
            LONG_PARTS_SOURCE,
            f'material_batch("batch-01", materials=[{_material_source("B")}])',
        ]
    )

    with pytest.raises(LongFormGenerationError, match="material IDs"):
        run_staged_generation(
            runner,
            plan_prompt="plan",
            material_prompt_template="{{batch_id}} {{material_specs}}",
            output_dir=tmp_path,
        )
    assert [call[0] for call in runner.calls] == ["plan", "material-batch-01"]
    assert not (tmp_path / "final.music.py").exists()


def test_plan_rejects_notes_and_unresolved_prompt_variables(tmp_path: Path) -> None:
    plan_with_notes = LONG_PARTS_SOURCE.replace(
        'material("A", duration_ms=44000, notes=[])',
        'material("A", duration_ms=44000, notes=['
        'note("a1", at_ms=0, duration_ms=1000, pitch=60, velocity=70)])',
    )
    runner = FakeRunner([plan_with_notes])
    with pytest.raises(LongFormGenerationError, match="must not contain notes"):
        run_staged_generation(
            runner,
            plan_prompt="plan",
            material_prompt_template="{{batch_id}} {{material_specs}}",
            output_dir=tmp_path,
        )

    unresolved = FakeRunner([LONG_PARTS_SOURCE])
    with pytest.raises(LongFormGenerationError, match="unresolved"):
        run_staged_generation(
            unresolved,
            plan_prompt="plan",
            material_prompt_template="{{batch_id}} {{material_specs}} {{unknown}}",
            output_dir=tmp_path / "other",
        )


def test_serializer_keeps_parts_instead_of_silently_flattening_them() -> None:
    plan = parse_composition(
        LONG_PARTS_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    serialized = composition_to_source(plan)

    assert "parts=[" in serialized
    assert "form=[" not in serialized
    assert json.loads(json.dumps(serialized)) == serialized


def test_serializer_preserves_phrases_and_material_derivation() -> None:
    plan = parse_composition(
        PHRASE_PARTS_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    serialized = composition_to_source(plan)

    assert "phrases=[" in serialized
    assert 'phrase("V1", role="variation", derived_from="S1"' in serialized
    assert 'material("B", duration_ms=44000, notes=[], pedals=[], derived_from="A")' in serialized
    assert (
        parse_composition(
            serialized,
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )
        == plan
    )


def test_serializer_preserves_note_voice_and_variation_kind() -> None:
    source = PHRASE_PARTS_SOURCE.replace(
        'phrase("V1", role="variation", derived_from="S1"',
        'phrase("V1", role="variation", derived_from="S1", variation_kind="rhythmic"',
    ).replace(
        'material("A", duration_ms=44000, notes=[])',
        'material("A", duration_ms=44000, notes=['
        'note("a1", 72, 80, 0, 400, voice="upper"), '
        'note("a2", 48, 65, 0, 900, voice="lower")])',
    )
    composition = parse_composition(
        source,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    serialized = composition_to_source(composition)

    assert 'variation_kind="rhythmic"' in serialized
    assert 'voice="upper"' in serialized
    assert (
        parse_composition(
            serialized,
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )
        == composition
    )


def test_derived_materials_are_generated_after_and_from_their_sources(tmp_path: Path) -> None:
    source_a = (
        'material("A", duration_ms=44000, notes=['
        'note("a1", at_ms=0, duration_ms=1000, pitch=60, velocity=70)], pedals=[])'
    )
    source_c = (
        'material("C", duration_ms=44000, notes=['
        'note("c1", at_ms=0, duration_ms=1000, pitch=67, velocity=80)], pedals=[])'
    )
    derived_b = (
        'material("B", duration_ms=44000, notes=['
        'note("b1", at_ms=500, duration_ms=1200, pitch=64, velocity=75)], '
        'pedals=[], derived_from="A")'
    )
    runner = FakeRunner(
        [
            PHRASE_PARTS_SOURCE,
            f'material_batch("batch-01", materials=[{source_a}, {source_c}])',
            f'material_batch("batch-02", materials=[{derived_b}])',
        ]
    )

    composition = run_staged_generation(
        runner,
        plan_prompt="plan",
        material_prompt_template="{{batch_id}} {{material_specs}}",
        output_dir=tmp_path,
    )

    assert [call[0] for call in runner.calls] == [
        "plan",
        "material-batch-01",
        "material-batch-02",
    ]
    assert '"source_material"' not in runner.calls[1][1]
    assert '"source_material"' in runner.calls[2][1]
    assert 'note(\\"a1\\"' in runner.calls[2][1]
    assert "dependency_materials" in runner.calls[2][2]
    assert composition.material_by_id["B"].derived_from == "A"


def test_exact_copy_is_rejected_as_a_variation(tmp_path: Path) -> None:
    source_a = (
        'material("A", duration_ms=44000, notes=['
        'note("a1", at_ms=0, duration_ms=1000, pitch=60, velocity=70)], pedals=[])'
    )
    source_c = (
        'material("C", duration_ms=44000, notes=['
        'note("c1", at_ms=0, duration_ms=1000, pitch=67, velocity=80)], pedals=[])'
    )
    copied_b = (
        'material("B", duration_ms=44000, notes=['
        'note("b1", at_ms=0, duration_ms=1000, pitch=60, velocity=70)], '
        'pedals=[], derived_from="A")'
    )
    runner = FakeRunner(
        [
            PHRASE_PARTS_SOURCE,
            f'material_batch("batch-01", materials=[{source_a}, {source_c}])',
            f'material_batch("batch-02", materials=[{copied_b}])',
        ]
    )

    with pytest.raises(LongFormGenerationError, match="exact copy"):
        run_staged_generation(
            runner,
            plan_prompt="plan",
            material_prompt_template="{{batch_id}} {{material_specs}}",
            output_dir=tmp_path,
        )


def test_exact_copy_can_be_deferred_only_for_the_typed_variation_repair() -> None:
    source = Material(
        "A",
        4_000,
        (Note("a1", 0, 800, 60, 70, "upper"),),
    )
    copied = replace(
        source,
        material_id="Av",
        notes=(replace(source.notes[0], event_id="av1"),),
        derived_from="A",
    )

    _validate_material_variations((copied,), {"A": source}, allow_exact_copy=True)
    with pytest.raises(LongFormGenerationError, match="exact copy"):
        _validate_material_variations((copied,), {"A": source})


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ('material_batch("wrong", materials=[])', "batch ID"),
        ('composition(title="x", form=[], materials=[])', "invalid material batch"),
        (
            'material_batch("batch-01", materials=['
            'material("A", duration_ms=43000, notes=[], pedals=[])])',
            "duration changed",
        ),
    ],
)
def test_invalid_batch_envelopes_stop_before_later_calls(
    tmp_path: Path, response: str, message: str
) -> None:
    runner = FakeRunner([LONG_PARTS_SOURCE, response])

    with pytest.raises(LongFormGenerationError, match=message):
        run_staged_generation(
            runner,
            plan_prompt="plan",
            material_prompt_template="{{batch_id}} {{material_specs}}",
            output_dir=tmp_path,
        )

    assert len(runner.calls) == 2


def test_duplicate_event_ids_across_batches_fail_final_assembly(tmp_path: Path) -> None:
    duplicate_b = _material_source("B").replace('"b1"', '"a1"')
    runner = FakeRunner(
        [
            LONG_PARTS_SOURCE,
            f'material_batch("batch-01", materials=[{_material_source("A")}])',
            f'material_batch("batch-02", materials=[{duplicate_b}])',
            f'material_batch("batch-03", materials=[{_material_source("C")}])',
        ]
    )

    with pytest.raises(LongFormGenerationError, match="event IDs"):
        run_staged_generation(
            runner,
            plan_prompt="plan",
            material_prompt_template="{{batch_id}} {{material_specs}}",
            output_dir=tmp_path,
        )


def test_flat_legacy_composition_serializer_keeps_the_legacy_form() -> None:
    serialized = composition_to_source(parse_composition(VALID_SOURCE))

    assert "form=[" in serialized
    assert "parts=[" not in serialized
    assert parse_composition(serialized) == parse_composition(VALID_SOURCE)


def test_missing_response_source_and_invalid_plan_are_explicit(tmp_path: Path) -> None:
    class MissingSourceRunner:
        def run(self, step_id, prompt, input_hashes=None):
            return {"intent_summary": "missing"}

    with pytest.raises(LongFormGenerationError, match="no composition_source"):
        run_staged_generation(
            MissingSourceRunner(),
            plan_prompt="plan",
            material_prompt_template="{{batch_id}} {{material_specs}}",
            output_dir=tmp_path,
        )

    invalid = FakeRunner(["composition("])
    with pytest.raises(LongFormGenerationError, match="invalid long-form plan"):
        run_staged_generation(
            invalid,
            plan_prompt="plan",
            material_prompt_template="{{batch_id}} {{material_specs}}",
            output_dir=tmp_path,
        )


INNER_STRUCTURE_SOURCE = """composition(
    title="inner",
    tonal_center=9,
    mode="minor",
    ending=tonic_hold(duration_ms=4000),
    parts=[
        part("P1", role="opening", energy=2, uses=[
            use("A", role="opening", energy=2), use("B", role="contrast", energy=2),
            use("A", role="return", energy=2), use("C", role="contrast", energy=2),
            use("A", role="return", energy=2), use("B", role="contrast", energy=2),
            use("A", role="return", energy=2),
        ]),
        part("P2", role="development", energy=3, uses=[
            use("D", role="contrast", energy=3), use("E", role="contrast", energy=3),
            use("D", role="return", energy=3), use("F", role="contrast", energy=3),
            use("D", role="return", energy=3), use("E", role="contrast", energy=3),
            use("D", role="return", energy=3), use("F", role="contrast", energy=3),
        ]),
        part("P3", role="development", energy=4, uses=[
            use("G", role="contrast", energy=4), use("H", role="contrast", energy=4),
            use("G", role="return", energy=4), use("I", role="contrast", energy=4),
            use("G", role="return", energy=4), use("H", role="contrast", energy=4),
            use("G", role="return", energy=4), use("I", role="contrast", energy=4),
        ]),
        part("P4", role="climax", energy=5, uses=[
            use("J", role="climax", energy=5), use("K", role="contrast", energy=4),
            use("J", role="return", energy=5), use("L", role="contrast", energy=4),
            use("J", role="return", energy=5), use("K", role="contrast", energy=4),
            use("J", role="return", energy=5), use("L", role="contrast", energy=4),
        ]),
        part("P5", role="return", energy=2, uses=[
            use("A", role="return", energy=2), use("B", role="contrast", energy=2),
            use("A", role="return", energy=2), use("C", role="contrast", energy=2),
            use("A", role="return", energy=2), use("B", role="contrast", energy=2),
            use("A", role="return", energy=2), use("C", role="contrast", energy=2),
        ]),
        part("P6", role="release", energy=1, uses=[
            use("M", role="release", energy=1), use("N", role="contrast", energy=1),
            use("M", role="return", energy=1), use("N", role="contrast", energy=1),
            use("M", role="return", energy=1),
        ]),
    ],
    materials=[
        material("A", duration_ms=4000, notes=[]), material("B", duration_ms=4000, notes=[]),
        material("C", duration_ms=4000, notes=[]), material("D", duration_ms=4000, notes=[]),
        material("E", duration_ms=4000, notes=[]), material("F", duration_ms=4000, notes=[]),
        material("G", duration_ms=4000, notes=[]), material("H", duration_ms=4000, notes=[]),
        material("I", duration_ms=4000, notes=[]), material("J", duration_ms=4000, notes=[]),
        material("K", duration_ms=4000, notes=[]), material("L", duration_ms=4000, notes=[]),
        material("M", duration_ms=4000, notes=[]), material("N", duration_ms=4000, notes=[]),
    ],
)"""


NATURAL_STRUCTURE_SOURCE = """composition(
    title="natural",
    tonal_center=9,
    mode="minor",
    ending=tonic_hold(duration_ms=4000),
    parts=[
        part("P1", role="opening", energy=2, uses=[
            use("A", role="opening", energy=2), use("B", role="contrast", energy=2),
            use("C", role="contrast", energy=2), use("A", role="return", energy=2),
            use("D", role="contrast", energy=2), use("T1", role="transition", energy=2),
        ]),
        part("P2", role="development", energy=3, uses=[
            use("E", role="contrast", energy=3), use("F", role="contrast", energy=3),
            use("G", role="contrast", energy=3), use("E", role="return", energy=3),
            use("H", role="contrast", energy=3), use("I", role="contrast", energy=3),
            use("T2", role="transition", energy=3),
        ]),
        part("P3", role="development", energy=4, uses=[
            use("J", role="contrast", energy=4), use("K", role="contrast", energy=4),
            use("L", role="contrast", energy=4), use("J", role="return", energy=4),
            use("M", role="contrast", energy=4), use("N", role="contrast", energy=4),
            use("T3", role="transition", energy=4),
        ]),
        part("P4", role="climax", energy=5, uses=[
            use("O", role="climax", energy=5), use("P", role="contrast", energy=4),
            use("Q", role="contrast", energy=4), use("O", role="return", energy=5),
            use("R", role="contrast", energy=4), use("T4", role="transition", energy=4),
        ]),
        part("P5", role="release", energy=2, uses=[
            use("A", role="return", energy=2), use("S", role="contrast", energy=2),
            use("B", role="contrast", energy=2), use("C", role="contrast", energy=2),
            use("A", role="return", energy=2), use("U", role="release", energy=2),
        ]),
    ],
    materials=[
        material("A", duration_ms=5500, notes=[]), material("B", duration_ms=5500, notes=[]),
        material("C", duration_ms=5500, notes=[]), material("D", duration_ms=5500, notes=[]),
        material("T1", duration_ms=5500, notes=[]), material("E", duration_ms=5500, notes=[]),
        material("F", duration_ms=5500, notes=[]), material("G", duration_ms=5500, notes=[]),
        material("H", duration_ms=5500, notes=[]), material("I", duration_ms=5500, notes=[]),
        material("T2", duration_ms=5500, notes=[]), material("J", duration_ms=5500, notes=[]),
        material("K", duration_ms=5500, notes=[]), material("L", duration_ms=5500, notes=[]),
        material("M", duration_ms=5500, notes=[]), material("N", duration_ms=5500, notes=[]),
        material("T3", duration_ms=5500, notes=[]), material("O", duration_ms=5500, notes=[]),
        material("P", duration_ms=5500, notes=[]), material("Q", duration_ms=5500, notes=[]),
        material("R", duration_ms=5500, notes=[]), material("T4", duration_ms=5500, notes=[]),
        material("S", duration_ms=5500, notes=[]), material("U", duration_ms=5500, notes=[]),
    ],
)"""


def test_inner_structure_rejects_one_long_material_per_part() -> None:
    flat = parse_composition(
        LONG_PARTS_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    with pytest.raises(LongFormGenerationError, match="at least 32 uses"):
        _validate_long_form_inner_structure(flat)

    structured = parse_composition(
        INNER_STRUCTURE_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    _validate_long_form_inner_structure(structured)


def test_phrase_structure_rejects_long_flat_or_relationless_plans() -> None:
    template = Path("prompts/long-form-plan.md").read_text(encoding="utf-8")
    example = template.split("```python\n", 1)[1].split("\n```", 1)[0]
    plan = parse_composition(
        example,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    _validate_long_form_phrase_structure(plan)

    long_a = replace(plan.material_by_id["A"], duration_ms=20_000)
    long_materials = tuple(
        long_a if material.material_id == "A" else material for material in plan.materials
    )
    with pytest.raises(LongFormGenerationError, match="duration"):
        _validate_long_form_phrase_structure(replace(plan, materials=long_materials))

    without_variations = replace(
        plan,
        phrases=tuple(
            replace(phrase, role="contrast", derived_from=None)
            if phrase.role == "variation"
            else phrase
            for phrase in plan.phrases
        ),
    )
    with pytest.raises(LongFormGenerationError, match="phrase variation"):
        _validate_long_form_phrase_structure(without_variations)

    without_contrast = replace(
        plan,
        phrases=tuple(
            replace(phrase, role="statement") if phrase.role == "contrast" else phrase
            for phrase in plan.phrases
        ),
    )
    with pytest.raises(LongFormGenerationError, match="intervening contrast"):
        _validate_long_form_phrase_structure(without_contrast)

    one_phrase_in_p1 = replace(
        plan,
        phrases=tuple(
            phrase for phrase in plan.phrases if phrase.part_id != "P1" or phrase.phrase_id == "A1"
        ),
    )
    with pytest.raises(LongFormGenerationError, match="at least two phrases"):
        _validate_long_form_phrase_structure(one_phrase_in_p1)


def test_natural_long_form_plan_requires_variety_and_transition_fills() -> None:
    natural = parse_composition(
        NATURAL_STRUCTURE_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    _validate_natural_long_form_plan(natural)

    missing_fill = parse_composition(
        NATURAL_STRUCTURE_SOURCE.replace(
            'use("T1", role="transition", energy=2)',
            'use("T1", role="contrast", energy=2)',
        ),
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    with pytest.raises(LongFormGenerationError, match="end with one transition"):
        _validate_natural_long_form_plan(missing_fill)

    repeated = parse_composition(
        NATURAL_STRUCTURE_SOURCE.replace(
            'use("B", role="contrast", energy=2)',
            'use("A", role="contrast", energy=2)',
            1,
        ),
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    with pytest.raises(LongFormGenerationError, match="at most twice per part"):
        _validate_natural_long_form_plan(repeated)

    low_variety = NATURAL_STRUCTURE_SOURCE
    for old, new in (("B", "A"), ("C", "A"), ("D", "A"), ("F", "E"), ("G", "E")):
        low_variety = low_variety.replace(f'use("{old}"', f'use("{new}"')
    low_variety_composition = parse_composition(
        low_variety,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    with pytest.raises(LongFormGenerationError, match="unique material ratio"):
        _validate_natural_long_form_plan(low_variety_composition)


def test_natural_materials_stay_diatonic_and_avoid_dense_dissonance() -> None:
    plan = parse_composition(
        NATURAL_STRUCTURE_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    consonant = Material(
        "A",
        5500,
        (
            Note("a1", 0, 500, 57, 80),
            Note("a2", 0, 500, 60, 80),
            Note("a3", 0, 500, 64, 80),
        ),
        (),
    )
    _validate_natural_materials(plan, (consonant,))

    chromatic = Material("A", 5500, (Note("x", 0, 500, 58, 80),), ())
    with pytest.raises(LongFormGenerationError, match="outside the allowed scale"):
        _validate_natural_materials(plan, (chromatic,))

    four_notes = Material(
        "A",
        5500,
        tuple(Note(f"f{index}", 0, 500, pitch, 80) for index, pitch in enumerate((57, 60, 64, 69))),
        (),
    )
    with pytest.raises(LongFormGenerationError, match="more than three simultaneous notes"):
        _validate_natural_materials(plan, (four_notes,))

    dissonant = Material(
        "A",
        5500,
        (Note("d1", 0, 500, 57, 80), Note("d2", 0, 500, 59, 80)),
        (),
    )
    with pytest.raises(LongFormGenerationError, match="forbidden simultaneous interval"):
        _validate_natural_materials(plan, (dissonant,))

    leading_tone_fill = Material("T1", 5500, (Note("t1", 0, 500, 68, 80),), ())
    _validate_natural_materials(plan, (leading_tone_fill,))


def test_out_of_scale_notes_are_removed_before_harmony_validation() -> None:
    plan = parse_composition(
        NATURAL_STRUCTURE_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    material = Material(
        "A",
        5_500,
        (
            Note("valid", 0, 500, 57, 80),
            Note("chromatic", 700, 500, 58, 82),
        ),
        (Pedal("on", 0, 127), Pedal("off", 5_500, 0)),
    )

    normalized = _normalize_out_of_scale_notes(plan, (material,))[0]

    assert [note.event_id for note in normalized.notes] == ["valid"]
    assert normalized.pedals == material.pedals
    _validate_natural_materials(plan, (normalized,))


def test_final_pedal_release_is_moved_to_material_boundary_only() -> None:
    material = Material(
        "A",
        5_500,
        (Note("n", 0, 5_400, 57, 80),),
        (
            Pedal("p1", 0, 127),
            Pedal("p2", 2_500, 0),
            Pedal("p3", 2_700, 127),
            Pedal("p4", 5_200, 0),
        ),
    )

    normalized = _normalize_final_pedal_releases((material,))[0]

    assert normalized.notes == material.notes
    assert normalized.pedals[:-1] == material.pedals[:-1]
    assert normalized.pedals[-1] == Pedal("p4", 5_500, 0)

    no_pedals = Material("B", 5_500, (), ())
    final_on = Material("C", 5_500, (), (Pedal("on", 0, 127),))
    assert _normalize_final_pedal_releases((no_pedals, final_on)) == (
        no_pedals,
        final_on,
    )


def test_transition_ending_keeps_the_leading_tone_from_a_forbidden_dyad() -> None:
    plan = parse_composition(
        NATURAL_STRUCTURE_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    fill = Material(
        "T1",
        5_500,
        (
            Note("lower", 5_100, 250, 62, 80),
            Note("leading", 5_100, 200, 68, 85),
            Note("earlier", 4_000, 300, 64, 75),
        ),
        (),
    )

    normalized = _normalize_forbidden_ending_chords(plan, (fill,))[0]

    assert [note.event_id for note in normalized.notes] == ["leading", "earlier"]
    _validate_natural_materials(plan, (normalized,))


def test_regular_material_ending_keeps_the_largest_consonant_subset() -> None:
    plan = parse_composition(
        NATURAL_STRUCTURE_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    material = Material(
        "J",
        5_500,
        (
            Note("root", 5_220, 280, 60, 80),
            Note("middle", 5_220, 280, 62, 90),
            Note("third", 5_220, 280, 64, 85),
            Note("earlier", 4_000, 300, 65, 75),
        ),
        (),
    )

    normalized = _normalize_forbidden_ending_chords(plan, (material,))[0]

    assert [note.event_id for note in normalized.notes] == ["root", "third", "earlier"]
    _validate_natural_materials(plan, (normalized,))


def test_forbidden_chords_are_normalized_at_intermediate_onsets_too() -> None:
    plan = parse_composition(
        NATURAL_STRUCTURE_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    material = Material(
        "J",
        5_500,
        (
            Note("root", 1_000, 400, 60, 80),
            Note("second", 1_000, 400, 62, 90),
            Note("third", 1_000, 400, 64, 85),
            Note("later", 4_000, 300, 65, 75),
        ),
        (),
    )

    normalized = _normalize_forbidden_ending_chords(plan, (material,))[0]

    assert [note.event_id for note in normalized.notes] == ["root", "third", "later"]
    _validate_natural_materials(plan, (normalized,))


def test_natural_contract_rejects_final_fill_reuse_and_adjacent_duplicates() -> None:
    plan = parse_composition(
        NATURAL_STRUCTURE_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    final_fill = replace(
        plan,
        form=(*plan.form[:-1], replace(plan.form[-1], role="transition")),
    )
    with pytest.raises(LongFormGenerationError, match="final part must not contain"):
        _validate_natural_long_form_plan(final_fill)

    reused_fill_form = list(plan.form)
    reused_fill_form[12] = replace(reused_fill_form[12], material_id="T1")
    with pytest.raises(LongFormGenerationError, match="transition materials must be unique"):
        _validate_natural_long_form_plan(replace(plan, form=tuple(reused_fill_form)))

    adjacent_form = list(plan.form)
    adjacent_form[2] = replace(adjacent_form[2], material_id="B")
    with pytest.raises(LongFormGenerationError, match="adjacent uses"):
        _validate_natural_long_form_plan(replace(plan, form=tuple(adjacent_form)))


def test_natural_harmony_requires_tonal_context_and_detects_large_chords() -> None:
    plan = parse_composition(
        NATURAL_STRUCTURE_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    no_tonal_context = replace(plan, tonal_center=None, mode=None)
    with pytest.raises(LongFormGenerationError, match="transition normalization"):
        _normalize_forbidden_ending_chords(no_tonal_context, ())
    with pytest.raises(LongFormGenerationError, match="harmony validation"):
        _validate_natural_materials(no_tonal_context, ())
    with pytest.raises(LongFormGenerationError, match="harmony generation"):
        _material_specs(no_tonal_context, (), natural_harmony=True)
    assert _pitches_form_forbidden_chord([57, 60, 64, 69]) is True


def test_natural_staged_generation_rejects_expanded_note_count_above_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = parse_composition(
        NATURAL_STRUCTURE_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    responses = [NATURAL_STRUCTURE_SOURCE]
    for index, batch in enumerate(_batches_for_plan(plan), 1):
        materials = ", ".join(
            f'material("{material.material_id}", duration_ms={material.duration_ms}, '
            "notes=[], pedals=[])"
            for material in batch
        )
        responses.append(f'material_batch("batch-{index:02d}", materials=[{materials}])')
    monkeypatch.setattr("llm_musical_composer.long_form_generation.MAX_NATURAL_NOTE_COUNT", 3)

    with pytest.raises(LongFormGenerationError, match="exceeds 3 notes"):
        run_staged_generation(
            FakeRunner(responses),
            plan_prompt="plan",
            material_prompt_template="{{batch_id}} {{material_specs}}",
            output_dir=tmp_path,
            require_inner_structure=True,
            require_naturalness=True,
        )


def test_material_specs_receive_numeric_style_contracts_without_names() -> None:
    composition = parse_composition(
        INNER_STRUCTURE_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    prompt_target = {
        "scope": "neighbor level ranges only",
        "axis_targets": {
            axis: {"level": {"range_low": low, "median": median, "range_high": high}}
            for axis, low, median, high in (
                ("density", 6.0, 10.0, 12.0),
                ("polyphony", 1.7, 1.9, 2.1),
                ("velocity", 95.0, 105.0, 115.0),
                ("register", 55.0, 58.0, 61.0),
            )
        },
    }

    specs = _material_specs(composition, composition.materials, prompt_target)

    assert all("generation_targets" in spec for spec in specs)
    assert all("target_note_count" in spec["generation_targets"] for spec in specs)
    assert "name" not in json.dumps(specs)
    assert (
        sum(
            spec["generation_targets"]["target_note_count"]["center"] * len(spec["usages"])
            for spec in specs
        )
        < 2_044
    )


def test_material_specs_describe_voice_and_typed_variation_contracts() -> None:
    template = Path("prompts/long-form-plan.md").read_text(encoding="utf-8")
    example = template.split("```python\n", 1)[1].split("\n```", 1)[0]
    composition = parse_composition(
        example,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    specs = _material_specs(composition, composition.materials)
    by_id = {spec["material_id"]: spec for spec in specs}

    assert by_id["A"]["voice_contract"]["required_voices"] == ["upper", "lower"]
    assert by_id["Av"]["variation_kind"] == "rhythmic"
    assert by_id["Cv"]["variation_kind"] == "textural"


def test_new_long_form_contract_requires_voice_and_variation_kind() -> None:
    template = Path("prompts/long-form-plan.md").read_text(encoding="utf-8")
    example = template.split("```python\n", 1)[1].split("\n```", 1)[0]
    plan = parse_composition(
        example,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    with pytest.raises(LongFormGenerationError, match="both voices"):
        _validate_voice_and_variations(plan)


def test_failed_variations_are_repaired_once_without_regenerating_other_materials() -> None:
    source = PHRASE_PARTS_SOURCE.replace(
        'phrase("V1", role="variation", derived_from="S1"',
        'phrase("V1", role="variation", derived_from="S1", variation_kind="rhythmic"',
    )
    plan = parse_composition(
        source,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    def voiced(material: Material, upper_onsets: tuple[int, ...]) -> Material:
        upper = tuple(
            Note(f"{material.material_id}-u-{index}", onset, 350, pitch, 80, "upper")
            for index, (onset, pitch) in enumerate(zip(upper_onsets, (72, 74, 76, 77), strict=True))
        )
        lower = tuple(
            Note(f"{material.material_id}-l-{index}", onset, 900, pitch, 68, "lower")
            for index, (onset, pitch) in enumerate(
                zip((0, 750, 1_500, 2_250), (48, 52, 50, 55), strict=True)
            )
        )
        return replace(material, notes=(*upper, *lower))

    base = voiced(plan.materials[0], (0, 500, 1_000, 1_500))
    invalid = replace(voiced(plan.materials[1], (0, 500, 1_000, 1_500)), derived_from="A")
    contrast = voiced(plan.materials[2], (0, 500, 1_000, 1_500))
    repaired = replace(voiced(plan.materials[1], (0, 350, 1_050, 1_800)), derived_from="A")
    repaired_source = _material_source_for_test(repaired)
    response = f'material_batch("repair-variations", materials=[{repaired_source}])'
    runner = FakeRunner([response])
    composition = replace(plan, materials=(base, invalid, contrast))

    accepted = _repair_failed_variations(
        runner,
        composition,
        {material.material_id: material for material in composition.materials},
        material_prompt_template="{{batch_id}} {{material_specs}}",
        base_input_hashes={"base": "hash"},
        natural_harmony=False,
    )

    assert accepted["A"] == base
    assert accepted["C"] == contrast
    assert accepted["B"] == repaired
    assert runner.calls[0][0] == "material-repair-variations"
    assert "repair_target" in runner.calls[0][1]


def test_rhythmic_warp_changes_timing_without_rewriting_pitch_or_lower_voice() -> None:
    material = Material(
        "V",
        5_500,
        (
            Note("u1", 0, 350, 72, 80, "upper"),
            Note("u2", 500, 350, 74, 80, "upper"),
            Note("u3", 1_000, 350, 76, 80, "upper"),
            Note("u4", 1_500, 350, 77, 80, "upper"),
            Note("l1", 0, 900, 48, 68, "lower"),
            Note("l2", 750, 900, 52, 68, "lower"),
        ),
        (),
        "A",
    )

    warped = _warp_rhythmic_material(material)

    original_upper = [note for note in material.notes if note.voice == "upper"]
    warped_upper = [note for note in warped.notes if note.voice == "upper"]
    assert [note.pitch for note in warped_upper] == [note.pitch for note in original_upper]
    assert [note.at_ms for note in warped_upper] != [note.at_ms for note in original_upper]
    assert [note for note in warped.notes if note.voice == "lower"] == [
        note for note in material.notes if note.voice == "lower"
    ]
    assert _warp_rhythmic_material(Material("short", 1_000, ())) == Material("short", 1_000, ())


def test_exact_derived_copy_is_repaired_without_changing_pitch_or_lower_voice() -> None:
    source = Material(
        "A",
        5_500,
        (
            Note("a-u1", 0, 350, 72, 80, "upper"),
            Note("a-u2", 500, 350, 74, 80, "upper"),
            Note("a-u3", 1_000, 350, 76, 80, "upper"),
            Note("a-u4", 1_500, 350, 77, 80, "upper"),
            Note("a-l1", 0, 900, 48, 68, "lower"),
        ),
    )
    copied = replace(
        source,
        material_id="Av",
        notes=tuple(replace(note, event_id=f"v-{note.event_id}") for note in source.notes),
        derived_from="A",
    )

    repaired = _repair_exact_material_copies({"A": source, "Av": copied})

    assert repaired["A"] == source
    assert [note.pitch for note in repaired["Av"].notes] == [note.pitch for note in copied.notes]
    assert [note for note in repaired["Av"].notes if note.voice == "lower"] == [
        note for note in copied.notes if note.voice == "lower"
    ]
    assert [note.at_ms for note in repaired["Av"].notes if note.voice == "upper"] != [
        note.at_ms for note in copied.notes if note.voice == "upper"
    ]
    _validate_material_variations((repaired["Av"],), repaired)


def test_rhythmic_contour_restore_changes_only_upper_pitch_sequence() -> None:
    source = Material(
        "A",
        5_500,
        (
            Note("a-u1", 0, 350, 72, 80, "upper"),
            Note("a-u2", 500, 350, 74, 81, "upper"),
            Note("a-u3", 1_000, 350, 76, 82, "upper"),
            Note("a-l1", 0, 900, 48, 68, "lower"),
        ),
    )
    variation = Material(
        "Av",
        5_500,
        (
            Note("av-u0", 0, 300, 67, 74, "upper"),
            Note("av-u1", 0, 300, 79, 76, "upper"),
            Note("av-u2", 350, 300, 67, 77, "upper"),
            Note("av-u3", 1_100, 450, 81, 78, "upper"),
            Note("av-u4", 1_600, 300, 83, 79, "upper"),
            Note("av-u5", 2_000, 300, 65, 75, "upper"),
            Note("av-l1", 100, 800, 43, 64, "lower"),
        ),
        derived_from="A",
    )

    restored = _restore_rhythmic_contour(source, variation)

    restored_upper = [note for note in restored.notes if note.voice == "upper"]
    assert [
        max(
            (note for note in restored_upper if note.at_ms == onset), key=lambda note: note.pitch
        ).pitch
        for onset in sorted({note.at_ms for note in restored_upper})
    ] == [72, 74, 76, 72, 74]
    assert [
        (note.event_id, note.at_ms, note.duration_ms, note.velocity)
        for note in restored.notes
        if note.voice == "upper"
    ] == [
        (note.event_id, note.at_ms, note.duration_ms, note.velocity)
        for note in variation.notes
        if note.voice == "upper"
    ]
    assert [note for note in restored.notes if note.voice == "lower"] == [
        note for note in variation.notes if note.voice == "lower"
    ]


def test_textural_restore_replaces_only_the_upper_motif() -> None:
    source = Material(
        "A",
        5_500,
        (
            Note("a-u1", 0, 350, 72, 80, "upper"),
            Note("a-u2", 500, 350, 74, 81, "upper"),
            Note("a-u3", 1_000, 350, 76, 82, "upper"),
            Note("a-l1", 0, 900, 48, 68, "lower"),
        ),
    )
    variation = Material(
        "At",
        4_000,
        (
            Note("at-u1", 0, 250, 81, 75, "upper"),
            Note("at-u2", 800, 250, 68, 76, "upper"),
            Note("at-l1", 100, 500, 43, 64, "lower"),
            Note("at-l2", 900, 700, 50, 66, "lower"),
        ),
        (Pedal("at-p1", 0, 127), Pedal("at-p2", 4_000, 0)),
        "A",
    )

    restored = _restore_textural_upper_motif(source, variation)

    scale = variation.duration_ms / source.duration_ms
    assert [
        (note.at_ms, note.duration_ms, note.pitch, note.velocity)
        for note in restored.notes
        if note.voice == "upper"
    ] == [
        (
            round(note.at_ms * scale),
            min(
                max(1, round(note.duration_ms * scale)),
                variation.duration_ms - round(note.at_ms * scale),
            ),
            note.pitch,
            note.velocity,
        )
        for note in source.notes
        if note.voice == "upper"
    ]
    assert [note for note in restored.notes if note.voice == "lower"] == [
        note for note in variation.notes if note.voice == "lower"
    ]
    assert restored.pedals == variation.pedals
    assert all(
        note.event_id.startswith("At-texture-u-")
        for note in restored.notes
        if note.voice == "upper"
    )
    for pitch in {note.pitch for note in restored.notes}:
        same_pitch = sorted(
            (note for note in restored.notes if note.pitch == pitch), key=lambda note: note.at_ms
        )
        assert all(
            first.at_ms + first.duration_ms <= second.at_ms
            for first, second in pairwise(same_pitch)
        )


def test_registral_restore_recovers_missing_upper_event_with_constant_shift() -> None:
    source = Material(
        "A",
        5_500,
        (
            Note("a-u1", 120, 520, 57, 94, "upper"),
            Note("a-u2", 700, 460, 60, 92, "upper"),
            Note("a-u3", 1_240, 500, 64, 97, "upper"),
            Note("a-l1", 0, 760, 45, 86, "lower"),
        ),
    )
    incomplete = Material(
        "Ar",
        5_500,
        (
            Note("ar-u1", 120, 520, 64, 94, "upper"),
            Note("ar-u3", 1_240, 500, 71, 97, "upper"),
            Note("ar-l1", 0, 760, 45, 86, "lower"),
        ),
        derived_from="A",
    )

    restored = _restore_registral_pattern(source, incomplete)

    upper = [note for note in restored.notes if note.voice == "upper"]
    assert [(note.at_ms, note.duration_ms, note.pitch) for note in upper] == [
        (120, 520, 64),
        (700, 460, 67),
        (1_240, 500, 71),
    ]
    assert [note for note in restored.notes if note.voice == "lower"] == [
        Note("ar-l1", 0, 760, 45, 86, "lower")
    ]

    scale_adjusted = _restore_registral_pattern(
        source,
        incomplete,
        allowed_pitch_classes=frozenset(set(range(12)) - {7}),
    )
    assert [
        note.pitch for note in scale_adjusted.notes if note.voice == "upper" and note.at_ms == 700
    ] == [66]


def test_failed_registral_repair_restores_pattern_before_final_validation() -> None:
    source = PHRASE_PARTS_SOURCE.replace(
        'phrase("V1", role="variation", derived_from="S1"',
        'phrase("V1", role="variation", derived_from="S1", variation_kind="registral"',
    )
    plan = parse_composition(
        source,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    def voiced(material: Material) -> Material:
        return replace(
            material,
            notes=(
                Note(f"{material.material_id}-u1", 0, 350, 72, 80, "upper"),
                Note(f"{material.material_id}-u2", 500, 350, 74, 80, "upper"),
                Note(f"{material.material_id}-u3", 1_000, 350, 76, 80, "upper"),
                Note(f"{material.material_id}-u4", 1_500, 350, 77, 80, "upper"),
                Note(f"{material.material_id}-l1", 0, 900, 48, 68, "lower"),
                Note(f"{material.material_id}-l2", 750, 900, 52, 68, "lower"),
            ),
        )

    base = voiced(plan.materials[0])
    incomplete = replace(
        voiced(plan.materials[1]),
        notes=(
            Note("B-u1", 0, 350, 79, 80, "upper"),
            Note("B-u4", 1_500, 350, 84, 80, "upper"),
            Note("B-l1", 0, 900, 48, 68, "lower"),
            Note("B-l2", 750, 900, 52, 68, "lower"),
        ),
        derived_from="A",
    )
    contrast = voiced(plan.materials[2])
    composition = replace(plan, materials=(base, incomplete, contrast))
    response = (
        f'material_batch("repair-variations", materials=[{_material_source_for_test(incomplete)}])'
    )

    accepted = _repair_failed_variations(
        FakeRunner([response]),
        composition,
        {material.material_id: material for material in composition.materials},
        material_prompt_template="{{batch_id}} {{material_specs}}",
        base_input_hashes={"base": "hash"},
        natural_harmony=False,
    )

    repaired_upper = [note for note in accepted["B"].notes if note.voice == "upper"]
    assert [(note.at_ms, note.pitch) for note in repaired_upper] == [
        (0, 79),
        (500, 81),
        (1_000, 83),
        (1_500, 84),
    ]


def test_natural_material_specs_reduce_density_and_describe_fill_destination() -> None:
    composition = parse_composition(
        NATURAL_STRUCTURE_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    prompt_target = {
        "axis_targets": {
            axis: {"level": {"range_low": low, "median": median, "range_high": high}}
            for axis, low, median, high in (
                ("density", 6.0, 10.0, 12.0),
                ("polyphony", 1.7, 1.9, 2.1),
                ("velocity", 95.0, 105.0, 115.0),
                ("register", 55.0, 58.0, 61.0),
            )
        }
    }

    specs = _material_specs(
        composition,
        composition.materials,
        prompt_target,
        density_scale=0.58,
        natural_harmony=True,
    )

    expanded_target = sum(
        spec["generation_targets"]["target_note_count"]["center"] * len(spec["usages"])
        for spec in specs
    )
    assert 700 <= expanded_target + 4 <= 950
    by_id = {spec["material_id"]: spec for spec in specs}
    assert by_id["A"]["tonal_context"]["allowed_pitch_classes"] == [0, 2, 4, 5, 7, 9, 11]
    assert by_id["T1"]["transition_to"] == {
        "part_id": "P2",
        "part_role": "development",
        "part_energy": 3,
    }


def test_related_motifs_are_kept_in_the_same_generation_batch() -> None:
    composition = parse_composition(
        INNER_STRUCTURE_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    batches = _batches_for_plan(composition)
    batch_sets = [{material.material_id for material in batch} for batch in batches]

    assert len(batches) <= 4
    for part in composition.parts:
        part_materials = {
            use.material_id for use in composition.form[part.start_use_index : part.end_use_index]
        }
        assert any(part_materials <= batch for batch in batch_sets)


def test_material_derivation_cannot_exceed_four_generation_batches() -> None:
    composition = parse_composition(
        LONG_PARTS_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    chain = tuple(
        Material(
            material_id,
            2_000,
            (),
            (),
            None if index == 0 else chr(ord("A") + index - 1),
        )
        for index, material_id in enumerate(("A", "B", "C", "D", "E"))
    )

    with pytest.raises(LongFormGenerationError, match="requires 5 batches"):
        _batches_for_plan(replace(composition, materials=chain))


def test_phrase_plan_splits_large_dependency_levels_without_mixing_children() -> None:
    template = Path("prompts/long-form-plan.md").read_text(encoding="utf-8")
    example = template.split("```python\n", 1)[1].split("\n```", 1)[0]
    composition = parse_composition(
        example,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    batches = _batches_for_plan(composition)

    assert len(batches) == 4
    assert all(len(batch) <= 8 for batch in batches)
    assert all(material.derived_from is None for batch in batches[:3] for material in batch)
    assert all(material.derived_from is not None for material in batches[3])


def test_prompt_renderer_does_not_confuse_nested_json_with_a_placeholder() -> None:
    nested_json = '{"outer":{"inner":1}}'

    assert _render_prompt("spec={{spec}}", {"spec": nested_json}) == f"spec={nested_json}"
