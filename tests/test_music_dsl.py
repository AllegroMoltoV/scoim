from __future__ import annotations

import pytest

from llm_musical_composer.music_dsl import (
    THREE_MINUTE_POLICY,
    DslError,
    parse_composition,
)

VALID_SOURCE = """composition(
    title="pilot",
    form=[use("A"), use("B"), use("A")],
    materials=[
        material("A", duration_ms=10000, notes=[
            note("a1", at_ms=0, duration_ms=800, pitch=60, velocity=70),
        ]),
        material("B", duration_ms=12000, notes=[
            note("b1", at_ms=100, duration_ms=900, pitch=64, velocity=72),
        ], pedals=[pedal("p1", at_ms=0, value=100)]),
    ],
)"""

ENDING_SOURCE = VALID_SOURCE.replace(
    'title="pilot",',
    'title="pilot", tonal_center=2, mode="minor", ending=tonic_hold(duration_ms=4000),',
)

CONTRACT_SOURCE = """composition(
    title="contract-pilot",
    form=[
        use("A", role="opening", energy=2),
        use("B", role="climax", energy=5),
        use("A", role="return", energy=2),
    ],
    materials=[
        material("A", duration_ms=14000, notes=[
            note("a1", at_ms=0, duration_ms=600, pitch=48, velocity=50),
            note("a2", at_ms=0, duration_ms=600, pitch=55, velocity=54),
            note("a3", at_ms=0, duration_ms=600, pitch=60, velocity=58),
            note("a4", at_ms=2500, duration_ms=600, pitch=50, velocity=51),
            note("a5", at_ms=2500, duration_ms=600, pitch=57, velocity=55),
            note("a6", at_ms=2500, duration_ms=600, pitch=62, velocity=59),
            note("a7", at_ms=5000, duration_ms=600, pitch=52, velocity=52),
            note("a8", at_ms=5000, duration_ms=600, pitch=59, velocity=56),
            note("a9", at_ms=5000, duration_ms=600, pitch=64, velocity=60),
            note("a10", at_ms=7500, duration_ms=600, pitch=50, velocity=50),
            note("a11", at_ms=7500, duration_ms=600, pitch=57, velocity=54),
            note("a12", at_ms=7500, duration_ms=600, pitch=62, velocity=58),
        ], pedals=[
            pedal("pa1", at_ms=0, value=127),
            pedal("pa2", at_ms=4500, value=0),
            pedal("pa3", at_ms=4700, value=127),
            pedal("pa4", at_ms=13700, value=0),
        ]),
        material("B", duration_ms=14000, notes=[
            note("b1", at_ms=0, duration_ms=300, pitch=72, velocity=80),
            note("b2", at_ms=600, duration_ms=300, pitch=74, velocity=82),
            note("b3", at_ms=1200, duration_ms=300, pitch=76, velocity=84),
            note("b4", at_ms=1800, duration_ms=300, pitch=77, velocity=86),
            note("b5", at_ms=2400, duration_ms=300, pitch=79, velocity=88),
            note("b6", at_ms=3000, duration_ms=300, pitch=81, velocity=90),
            note("b7", at_ms=3600, duration_ms=300, pitch=79, velocity=88),
            note("b8", at_ms=4200, duration_ms=300, pitch=77, velocity=86),
            note("b9", at_ms=4800, duration_ms=300, pitch=76, velocity=84),
            note("b10", at_ms=5400, duration_ms=300, pitch=74, velocity=82),
            note("b11", at_ms=6000, duration_ms=300, pitch=72, velocity=80),
            note("b12", at_ms=6600, duration_ms=300, pitch=74, velocity=82),
            note("b13", at_ms=7200, duration_ms=300, pitch=76, velocity=84),
            note("b14", at_ms=7800, duration_ms=300, pitch=77, velocity=86),
            note("b15", at_ms=8400, duration_ms=300, pitch=79, velocity=88),
            note("b16", at_ms=9000, duration_ms=300, pitch=81, velocity=90),
        ], pedals=[
            pedal("pb1", at_ms=0, value=127),
            pedal("pb2", at_ms=4500, value=0),
            pedal("pb3", at_ms=4700, value=127),
            pedal("pb4", at_ms=13700, value=0),
        ]),
    ],
)"""

LEGACY_CONTRACT_SOURCE = CONTRACT_SOURCE.replace(
    "energy=2)", 'energy=2, attack_style="clustered")'
).replace("energy=5)", 'energy=5, attack_style="distributed")')

CONTRACT_ENDING_SOURCE = CONTRACT_SOURCE.replace(
    'title="contract-pilot",',
    'title="contract-pilot", tonal_center=2, mode="minor", ending=tonic_hold(duration_ms=4000),',
)

PARTS_SOURCE = VALID_SOURCE.replace(
    'form=[use("A"), use("B"), use("A")],',
    """parts=[
        part("P1", role="opening", energy=2, uses=[use("A")]),
        part("P2", role="climax", energy=5, uses=[use("B")]),
        part("P3", role="return", energy=2, uses=[use("A")]),
    ],""",
)

LONG_PARTS_SOURCE = """composition(
    title="long",
    tonal_center=9,
    mode="minor",
    ending=tonic_hold(duration_ms=4000),
    parts=[
        part("P1", role="opening", energy=2, uses=[use("A", role="opening", energy=2)]),
        part("P2", role="development", energy=3, uses=[use("B", role="contrast", energy=3)]),
        part("P3", role="climax", energy=5, uses=[use("C", role="climax", energy=5)]),
        part("P4", role="return", energy=2, uses=[use("A", role="return", energy=2)]),
    ],
    materials=[
        material("A", duration_ms=44000, notes=[]),
        material("B", duration_ms=44000, notes=[]),
        material("C", duration_ms=44000, notes=[]),
    ],
)"""

PHRASE_PARTS_SOURCE = """composition(
    title="phrases",
    tonal_center=9,
    mode="minor",
    ending=tonic_hold(duration_ms=4000),
    parts=[
        part("P1", role="opening", energy=2, phrases=[
            phrase("S1", role="statement", uses=[
                use("A", role="opening", energy=2),
            ]),
        ]),
        part("P2", role="development", energy=3, phrases=[
            phrase("V1", role="variation", derived_from="S1", uses=[
                use("B", role="contrast", energy=3),
            ]),
        ]),
        part("P3", role="climax", energy=5, phrases=[
            phrase("C1", role="contrast", uses=[
                use("C", role="climax", energy=5),
            ]),
        ]),
        part("P4", role="return", energy=2, phrases=[
            phrase("R1", role="return", derived_from="S1", uses=[
                use("A", role="return", energy=2),
            ]),
        ]),
    ],
    materials=[
        material("A", duration_ms=44000, notes=[]),
        material("B", duration_ms=44000, notes=[], derived_from="A"),
        material("C", duration_ms=44000, notes=[]),
    ],
)"""


def test_parse_allowed_two_layer_composition() -> None:
    composition = parse_composition(VALID_SOURCE)

    assert composition.title == "pilot"
    assert [section.material_id for section in composition.form] == ["A", "B", "A"]
    assert composition.duration_ms == 32000
    assert composition.note_count == 3


def test_compact_note_and_cc64_alias_are_safe_dsl_syntax() -> None:
    compact = VALID_SOURCE.replace(
        'note("b1", at_ms=100, duration_ms=900, pitch=64, velocity=72)',
        'note("b1", 64, 72, 100, 900)',
    ).replace('pedal("p1", at_ms=0, value=100)', 'cc64("p1", at_ms=0, value=100)')

    composition = parse_composition(compact)

    note = composition.material_by_id["B"].notes[0]
    assert (note.pitch, note.velocity, note.at_ms, note.duration_ms) == (64, 72, 100, 900)
    assert composition.material_by_id["B"].pedals[0].value == 100


def test_note_voice_is_optional_and_restricted_to_audible_layers() -> None:
    voiced = VALID_SOURCE.replace(
        'note("a1", at_ms=0, duration_ms=800, pitch=60, velocity=70)',
        'note("a1", at_ms=0, duration_ms=800, pitch=60, velocity=70, voice="upper")',
    )

    assert parse_composition(VALID_SOURCE).material_by_id["A"].notes[0].voice is None
    assert parse_composition(voiced).material_by_id["A"].notes[0].voice == "upper"

    with pytest.raises(DslError, match="voice"):
        parse_composition(voiced.replace('voice="upper"', 'voice="right_hand"'))


def test_parts_add_one_layer_without_duplicating_the_flat_form() -> None:
    flat = parse_composition(VALID_SOURCE)
    hierarchical = parse_composition(PARTS_SOURCE)

    assert hierarchical.form == flat.form
    ranges = [
        (part.part_id, part.start_use_index, part.end_use_index) for part in hierarchical.parts
    ]
    assert ranges == [
        ("P1", 0, 1),
        ("P2", 1, 2),
        ("P3", 2, 3),
    ]
    assert hierarchical.duration_ms == flat.duration_ms


def test_phrases_add_a_middle_layer_and_preserve_the_flat_form() -> None:
    composition = parse_composition(
        PHRASE_PARTS_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    assert [phrase.phrase_id for phrase in composition.phrases] == ["S1", "V1", "C1", "R1"]
    assert [phrase.part_id for phrase in composition.phrases] == ["P1", "P2", "P3", "P4"]
    assert [(phrase.start_use_index, phrase.end_use_index) for phrase in composition.phrases] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    assert [use.material_id for use in composition.form] == ["A", "B", "C", "A"]
    assert composition.material_by_id["B"].derived_from == "A"


def test_variation_kind_is_optional_for_legacy_but_restricted_to_variations() -> None:
    rhythmic = PHRASE_PARTS_SOURCE.replace(
        'phrase("V1", role="variation", derived_from="S1"',
        'phrase("V1", role="variation", derived_from="S1", variation_kind="rhythmic"',
    )

    assert (
        parse_composition(
            PHRASE_PARTS_SOURCE,
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )
        .phrases[1]
        .variation_kind
        is None
    )
    assert (
        parse_composition(
            rhythmic,
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )
        .phrases[1]
        .variation_kind
        == "rhythmic"
    )

    with pytest.raises(DslError, match="variation_kind"):
        parse_composition(
            rhythmic.replace('variation_kind="rhythmic"', 'variation_kind="unknown"'),
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )
    with pytest.raises(DslError, match="variation phrase"):
        parse_composition(
            PHRASE_PARTS_SOURCE.replace(
                'phrase("S1", role="statement"',
                'phrase("S1", role="statement", variation_kind="rhythmic"',
            ),
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            PHRASE_PARTS_SOURCE.replace(
                'part("P1", role="opening", energy=2, phrases=[',
                'part("P1", role="opening", energy=2, uses=[use("A")], phrases=[',
            ),
            "exactly one of uses or phrases",
        ),
        (PHRASE_PARTS_SOURCE.replace('phrase("S1"', 'phrase(""'), "phrase IDs"),
        (
            PHRASE_PARTS_SOURCE.replace(
                'phrase("V1", role="variation", derived_from="S1"',
                'phrase("V1", role="variation"',
            ),
            "requires derived_from",
        ),
        (
            PHRASE_PARTS_SOURCE.replace('derived_from="S1"', 'derived_from="C1"', 1),
            "earlier phrase",
        ),
        (
            PHRASE_PARTS_SOURCE.replace('derived_from="A")', 'derived_from="C")'),
            "earlier material",
        ),
        (
            PHRASE_PARTS_SOURCE.replace('derived_from="A")', 'derived_from="missing")'),
            "unknown material",
        ),
        (
            PHRASE_PARTS_SOURCE.replace(
                'material("B", duration_ms=44000, notes=[], derived_from="A")',
                'material("D", duration_ms=44000, notes=[]),\n'
                '        material("B", duration_ms=44000, notes=[], derived_from="D")',
            ),
            "source phrase",
        ),
    ],
)
def test_rejects_invalid_phrase_and_material_relations(source: str, message: str) -> None:
    with pytest.raises(DslError, match=message):
        parse_composition(
            source,
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )


def test_hierarchical_climax_may_repeat_the_highest_energy_inside_one_part() -> None:
    repeated_climax = LONG_PARTS_SOURCE.replace(
        'part("P3", role="climax", energy=5, uses=[use("C", role="climax", energy=5)])',
        'part("P3", role="climax", energy=5, uses=['
        'use("C", role="climax", energy=5), use("C", role="return", energy=5)])',
    ).replace('material("C", duration_ms=44000', 'material("C", duration_ms=22000')

    composition = parse_composition(
        repeated_climax,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    assert [use.energy for use in composition.form[2:4]] == [5, 5]


def test_hierarchical_highest_energy_outside_climax_part_is_rejected() -> None:
    with pytest.raises(DslError, match="highest energy uses must stay inside the climax part"):
        parse_composition(
            LONG_PARTS_SOURCE.replace(
                'use("B", role="contrast", energy=3)',
                'use("B", role="contrast", energy=5)',
            ),
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )


def test_transition_is_an_allowed_use_role() -> None:
    with_transition = LONG_PARTS_SOURCE.replace(
        'use("B", role="contrast", energy=3)',
        'use("B", role="transition", energy=3)',
    )

    composition = parse_composition(
        with_transition,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    assert composition.form[1].role == "transition"


def test_form_and_parts_are_exclusive_and_parts_must_be_valid() -> None:
    with pytest.raises(DslError, match="exactly one of form or parts"):
        parse_composition(PARTS_SOURCE.replace("parts=[", 'form=[use("A")], parts=['))
    with pytest.raises(DslError, match="exactly one of form or parts"):
        parse_composition(VALID_SOURCE.replace('form=[use("A"), use("B"), use("A")],', ""))
    with pytest.raises(DslError, match="part IDs"):
        parse_composition(PARTS_SOURCE.replace('part("P3"', 'part("P2"'))
    with pytest.raises(DslError, match="must not be empty"):
        parse_composition(PARTS_SOURCE.replace('uses=[use("B")]', "uses=[]"))


def test_three_minute_policy_is_opt_in_and_exact() -> None:
    composition = parse_composition(
        LONG_PARTS_SOURCE,
        policy=THREE_MINUTE_POLICY,
        require_parts=True,
        require_section_contract=True,
    )

    assert composition.duration_ms == 180_000
    assert len(composition.parts) == 4
    with pytest.raises(DslError, match="expanded duration"):
        parse_composition(LONG_PARTS_SOURCE, require_parts=True)
    with pytest.raises(DslError, match="180000"):
        parse_composition(
            LONG_PARTS_SOURCE.replace("duration_ms=44000", "duration_ms=43999", 1),
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('role="opening"', 'role="unknown"', "part role"),
        ("energy=2", "energy=0", "part energy"),
        ('part("P1", role="opening"', 'part("P1", role="return"', "first part"),
        ('part("P3", role="climax"', 'part("P3", role="development"', "one climax"),
        ('part("P1", role="opening"', 'part("P1", role="climax"', "first part"),
        ('part("P3", role="climax"', 'part("P3", role="return"', "one climax"),
        ('part("P4", role="return"', 'part("P4", role="climax"', "one climax"),
        ("energy=3", "energy=5", "unique highest"),
        ('use("C", role="climax"', 'use("C", role="contrast"', "one climax"),
    ],
)
def test_rejects_invalid_long_form_part_contract(old: str, new: str, message: str) -> None:
    with pytest.raises(DslError, match=message):
        parse_composition(
            LONG_PARTS_SOURCE.replace(old, new, 1),
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )


def test_long_policy_requires_at_least_three_parts() -> None:
    shortened = (
        LONG_PARTS_SOURCE.replace(
            '        part("P2", role="development", energy=3, '
            'uses=[use("B", role="contrast", energy=3)]),\n',
            "",
        )
        .replace(
            '        part("P4", role="return", energy=2, '
            'uses=[use("A", role="return", energy=2)]),\n',
            "",
        )
        .replace(
            'uses=[use("A", role="opening", energy=2)]',
            'uses=[use("A", role="opening", energy=2), use("A", role="return", energy=2)]',
        )
    )
    with pytest.raises(DslError, match="at least 3"):
        parse_composition(
            shortened,
            policy=THREE_MINUTE_POLICY,
            require_parts=True,
            require_section_contract=True,
        )


def test_parse_tonic_ending_contract() -> None:
    composition = parse_composition(ENDING_SOURCE)

    assert composition.tonal_center == 2
    assert composition.mode == "minor"
    assert composition.ending is not None
    assert composition.ending.duration_ms == 4000
    assert composition.body_duration_ms == 32000
    assert composition.duration_ms == 36000
    assert composition.note_count == 7


def test_parse_required_section_contract() -> None:
    composition = parse_composition(CONTRACT_ENDING_SOURCE, require_section_contract=True)

    assert [section.role for section in composition.form] == ["opening", "climax", "return"]
    assert [section.energy for section in composition.form] == [2, 5, 2]
    assert [section.attack_style for section in composition.form] == [None, None, None]


def test_parse_legacy_attack_style_without_requiring_it_for_new_contract() -> None:
    legacy = LEGACY_CONTRACT_SOURCE.replace(
        'title="contract-pilot",',
        'title="contract-pilot", tonal_center=2, mode="minor", '
        "ending=tonic_hold(duration_ms=4000),",
    )

    composition = parse_composition(legacy, require_section_contract=True)

    assert [section.attack_style for section in composition.form] == [
        "clustered",
        "distributed",
        "clustered",
    ]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (ENDING_SOURCE, "section contract is required"),
        (
            CONTRACT_ENDING_SOURCE.replace(", energy=2", "", 1),
            "specified together",
        ),
        (CONTRACT_ENDING_SOURCE.replace('role="opening"', 'role="verse"'), "role"),
        (CONTRACT_ENDING_SOURCE.replace("energy=2", "energy=0", 1), "energy"),
        (CONTRACT_ENDING_SOURCE.replace('role="opening"', 'role="return"'), "opening"),
        (CONTRACT_ENDING_SOURCE.replace('role="climax"', 'role="contrast"'), "one climax"),
        (CONTRACT_ENDING_SOURCE.replace("energy=5", "energy=2"), "highest energy"),
        (
            CONTRACT_ENDING_SOURCE.replace(
                "energy=2),\n    ],",
                "energy=3),\n    ],",
            ),
            "reused material",
        ),
    ],
)
def test_rejects_invalid_required_section_contract(source: str, message: str) -> None:
    with pytest.raises(DslError, match=message):
        parse_composition(source, require_section_contract=True)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("tonal_center=2", "tonal_center=12", "tonal_center"),
        ('mode="minor"', 'mode="dorian"', "mode"),
        ("duration_ms=4000", "duration_ms=2999", "ending duration"),
        ("duration_ms=4000", "duration_ms=6001", "ending duration"),
        (', mode="minor"', "", "specified together"),
        (", ending=tonic_hold(duration_ms=4000)", "", "specified together"),
    ],
)
def test_rejects_invalid_tonic_ending_contract(old: str, new: str, message: str) -> None:
    with pytest.raises(DslError, match=message):
        parse_composition(ENDING_SOURCE.replace(old, new))


@pytest.mark.parametrize(
    "source",
    [
        '__import__("os").system("echo unsafe")',
        'composition(title="x", form=[], materials=open("x"))',
        'composition(title="x", form=[x for x in []], materials=[])',
        'composition(title="x", form=[], materials=[unknown()])',
        'composition(title="x", form=[], materials=[]) + 1',
    ],
)
def test_rejects_python_outside_the_dsl(source: str) -> None:
    with pytest.raises(DslError):
        parse_composition(source)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('use("A"), use("B"), use("A")', 'use("A"), use("B"), use("C")', "unknown"),
        ("pitch=60", "pitch=109", "pitch"),
        ("duration_ms=800", "duration_ms=11000", "material"),
        ('note("a1"', 'note("b1"', "unique"),
    ],
)
def test_rejects_semantically_invalid_compositions(old: str, new: str, message: str) -> None:
    with pytest.raises(DslError, match=message):
        parse_composition(VALID_SOURCE.replace(old, new))


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (VALID_SOURCE.replace('title="pilot"', 'title=""'), "title"),
        (VALID_SOURCE.replace('use("A"), use("B"), use("A")', 'use("A"), use("A")'), "three"),
        (VALID_SOURCE.replace('material("B"', 'material("A"'), "unique"),
        (
            VALID_SOURCE.replace(
                'use("A"), use("B"), use("A")', 'use("A"), use("B"), use("B")'
            ).replace('material("B"', 'material("C"'),
            "unknown",
        ),
        (VALID_SOURCE.replace("at_ms=0, duration_ms=800", "at_ms=-1, duration_ms=800"), "literal"),
        (VALID_SOURCE.replace("velocity=70", "velocity=128"), "velocity"),
        (
            VALID_SOURCE.replace(
                'pedal("p1", at_ms=0, value=100)', 'pedal("p1", at_ms=0, value=128)'
            ),
            "pedal",
        ),
        (
            VALID_SOURCE.replace(
                'pedal("p1", at_ms=0, value=100)', 'pedal("p1", at_ms=13000, value=100)'
            ),
            "pedal",
        ),
        (
            VALID_SOURCE.replace(
                'material("A", duration_ms=10000', 'material("A", duration_ms=30000'
            ),
            "60000",
        ),
        (
            VALID_SOURCE.replace(
                'form=[use("A"), use("B"), use("A")]',
                'form=[use("A"), use("B"), use("A"), use("B"), use("A"), use("B")]',
            ),
            "60000",
        ),
    ],
)
def test_more_semantic_rejections(source: str, message: str) -> None:
    with pytest.raises(DslError, match=message):
        parse_composition(source)


def test_rejects_overlapping_same_pitch() -> None:
    source = VALID_SOURCE.replace(
        'note("a1", at_ms=0, duration_ms=800, pitch=60, velocity=70),',
        'note("a1", at_ms=0, duration_ms=800, pitch=60, velocity=70),\n'
        'note("a2", at_ms=400, duration_ms=800, pitch=60, velocity=70),',
    )
    with pytest.raises(DslError, match="overlapping"):
        parse_composition(source)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("composition(**{})", "expansion"),
        (VALID_SOURCE.replace('use("A")', 'use("A", "B")', 1), "positional"),
        (VALID_SOURCE.replace('note("a1",', 'note("a1", event_id="x",'), "duplicate"),
        (VALID_SOURCE.replace('title="pilot",', 'title="pilot", extra=1,'), "unknown argument"),
        (VALID_SOURCE.replace('title="pilot",', ""), "missing argument"),
        (VALID_SOURCE.replace('form=[use("A"), use("B"), use("A")]', 'form="A-B-A"'), "list"),
        ("composition(", "invalid syntax"),
    ],
)
def test_rejects_malformed_dsl_arguments(source: str, message: str) -> None:
    with pytest.raises(DslError, match=message):
        parse_composition(source)
