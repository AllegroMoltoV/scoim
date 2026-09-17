from __future__ import annotations

from dataclasses import replace

import pytest

from llm_musical_composer.composition_ir import Material, Note
from llm_musical_composer.long_form_generation import (
    LongFormGenerationError,
    _repair_failed_variations,
)
from llm_musical_composer.music_dsl import THREE_MINUTE_POLICY, parse_composition
from tests.test_music_dsl import PHRASE_PARTS_SOURCE


class Runner:
    def __init__(self, source: str) -> None:
        self.source = source
        self.calls = 0

    def run(self, step_id, prompt, input_hashes=None):
        self.calls += 1
        return {"composition_source": self.source, "intent_summary": "test"}


def _notes(
    material_id: str,
    upper_onsets: tuple[int, ...],
    *,
    upper_pitches: tuple[int, ...] = (72, 74, 76, 77),
) -> tuple[Note, ...]:
    return (
        *(
            Note(f"{material_id}-u-{index}", onset, 350, pitch, 80, "upper")
            for index, (onset, pitch) in enumerate(zip(upper_onsets, upper_pitches, strict=True))
        ),
        *(
            Note(f"{material_id}-l-{index}", onset, 900, pitch, 68, "lower")
            for index, (onset, pitch) in enumerate(
                zip((0, 750, 1_500, 2_250), (48, 52, 50, 55), strict=True)
            )
        ),
    )


def _composition(kind: str = "rhythmic"):
    source = PHRASE_PARTS_SOURCE.replace(
        'phrase("V1", role="variation", derived_from="S1"',
        f'phrase("V1", role="variation", derived_from="S1", variation_kind="{kind}"',
    )
    plan = parse_composition(
        source,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )
    materials = (
        replace(plan.materials[0], notes=_notes("A", (0, 500, 1_000, 1_500))),
        replace(plan.materials[1], notes=_notes("B", (0, 500, 1_000, 1_500))),
        replace(plan.materials[2], notes=_notes("C", (0, 500, 1_000, 1_500))),
    )
    composition = replace(plan, materials=materials)
    return composition, {material.material_id: material for material in materials}


def _source(material: Material, *, batch_id: str = "repair-variations") -> str:
    notes = ", ".join(
        f'note("{note.event_id}", {note.pitch}, {note.velocity}, {note.at_ms}, '
        f'{note.duration_ms}, voice="{note.voice}")'
        for note in material.notes
    )
    derived = (
        f', derived_from="{material.derived_from}"' if material.derived_from is not None else ""
    )
    body = (
        f'material("{material.material_id}", duration_ms={material.duration_ms}, '
        f"notes=[{notes}], pedals=[]{derived})"
    )
    return f'material_batch("{batch_id}", materials=[{body}])'


def _repair(
    runner: Runner,
    composition,
    accepted,
    *,
    natural_harmony: bool = False,
):
    return _repair_failed_variations(
        runner,
        composition,
        accepted,
        material_prompt_template="{{batch_id}} {{material_specs}}",
        base_input_hashes={},
        natural_harmony=natural_harmony,
    )


def test_repair_returns_without_a_call_when_variations_already_pass() -> None:
    composition, accepted = _composition()
    changed = replace(
        accepted["B"],
        notes=_notes("B", (0, 350, 1_050, 1_800)),
    )
    composition = replace(
        composition,
        materials=(accepted["A"], changed, accepted["C"]),
    )
    runner = Runner("")

    result = _repair(runner, composition, {**accepted, "B": changed})

    assert result["B"] == changed
    assert runner.calls == 0


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("wrong_batch", "batch ID"),
        ("wrong_id", "exactly match"),
        ("wrong_duration", "duration or derivation"),
    ],
)
def test_repair_rejects_scope_and_contract_changes(response: str, message: str) -> None:
    composition, accepted = _composition()
    repaired = replace(
        accepted["B"],
        notes=_notes("B", (0, 350, 1_050, 1_800)),
    )
    if response == "wrong_batch":
        source = _source(repaired, batch_id="wrong")
    elif response == "wrong_id":
        source = _source(replace(repaired, material_id="C"))
    else:
        source = _source(replace(repaired, duration_ms=43_000))

    with pytest.raises(LongFormGenerationError, match=message):
        _repair(Runner(source), composition, accepted)


def test_repair_runs_natural_normalization_and_rhythm_fallback() -> None:
    composition, accepted = _composition()
    near_copy = replace(
        accepted["B"],
        notes=tuple(
            replace(note, duration_ms=note.duration_ms + 10) if note.voice == "upper" else note
            for note in accepted["B"].notes
        ),
    )

    result = _repair(Runner(_source(near_copy)), composition, accepted, natural_harmony=True)

    assert [note.pitch for note in result["B"].notes] == [note.pitch for note in near_copy.notes]
    assert [note.at_ms for note in result["B"].notes if note.voice == "upper"] != [
        note.at_ms for note in near_copy.notes if note.voice == "upper"
    ]


def test_repair_does_not_warp_textural_failure_without_lower_change() -> None:
    composition, accepted = _composition("textural")
    upper_pitches = (72, 74, 76, 77)
    upper_onsets = (0, 500, 1_000, 1_500)
    invalid = replace(
        accepted["B"],
        notes=_notes("B", upper_onsets, upper_pitches=upper_pitches),
    )
    invalid = replace(
        invalid,
        notes=(replace(invalid.notes[0], velocity=81), *invalid.notes[1:]),
    )

    with pytest.raises(LongFormGenerationError, match="change the lower layer"):
        _repair(Runner(_source(invalid)), composition, accepted)


def test_repair_restores_broken_rhythmic_contour_without_changing_rhythm() -> None:
    composition, accepted = _composition("rhythmic")
    upper_onsets = (0, 350, 1_050, 1_800)
    invalid = replace(
        accepted["B"],
        notes=_notes("B", upper_onsets, upper_pitches=(72, 67, 79, 65)),
    )

    repaired = _repair(Runner(_source(invalid)), composition, accepted)

    repaired_upper = [note for note in repaired["B"].notes if note.voice == "upper"]
    assert [note.pitch for note in repaired_upper] == [72, 74, 76, 77]
    assert [note.at_ms for note in repaired_upper] == list(upper_onsets)
