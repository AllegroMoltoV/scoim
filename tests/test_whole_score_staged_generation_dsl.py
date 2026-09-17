from __future__ import annotations

import pytest

from llm_musical_composer.staged_material_pilot import (
    HarmonicDraft,
    HarmonicEventDraft,
    MelodyDraft,
    MelodyEventDraft,
    TextureDraft,
    TextureEventDraft,
)
from llm_musical_composer.whole_score_staged_generation import (
    PerformanceOccurrenceDraftV0,
    WholeHarmonicMaterialDraftV0,
)
from llm_musical_composer.whole_score_staged_generation_dsl import (
    WholeScoreCollectionDslError,
    dump_harmonic_collection,
    dump_melody_collection,
    dump_performance_collection,
    dump_texture_collection,
    parse_harmonic_collection,
    parse_melody_collection,
    parse_performance_collection,
    parse_texture_collection,
)


@pytest.mark.parametrize(
    ("value", "dump", "parse"),
    (
        (
            (
                WholeHarmonicMaterialDraftV0(
                    12,
                    HarmonicDraft((HarmonicEventDraft(0, 12, 0, "major"),)),
                ),
                WholeHarmonicMaterialDraftV0(
                    6,
                    HarmonicDraft((HarmonicEventDraft(0, 6, 7, "major"),)),
                ),
            ),
            dump_harmonic_collection,
            parse_harmonic_collection,
        ),
        (
            (
                MelodyDraft("upper", (MelodyEventDraft(0, 3, 60, ("tenuto",)),)),
                MelodyDraft("lower", (MelodyEventDraft(0, 2, 48),)),
            ),
            dump_melody_collection,
            parse_melody_collection,
        ),
        (
            (
                TextureDraft((TextureEventDraft(0, 0, 6, "root", "bass"),)),
                TextureDraft((TextureEventDraft(0, 0, 3, "fifth", "low"),)),
            ),
            dump_texture_collection,
            parse_texture_collection,
        ),
        (
            (
                PerformanceOccurrenceDraftV0(
                    "savor",
                    "subtle",
                    "shape",
                    "legato",
                    "rolled",
                    "harmony_legato",
                ),
                PerformanceOccurrenceDraftV0(
                    "release",
                    "subtle",
                    "release",
                    "legato",
                    "aligned",
                    "harmony_legato",
                ),
            ),
            dump_performance_collection,
            parse_performance_collection,
        ),
    ),
)
def test_collection_dsl_roundtrips_deterministically(value, dump, parse) -> None:
    source = dump(value)

    assert parse(source) == value
    assert dump(parse(source)) == source


@pytest.mark.parametrize(
    ("parse", "source"),
    (
        (parse_harmonic_collection, "unknown(materials=[] )"),
        (parse_harmonic_collection, "harmonic_collection([])"),
        (
            parse_harmonic_collection,
            "harmonic_collection(materials=[], material_ids=['forbidden'])",
        ),
        (
            parse_melody_collection,
            "melody_collection(materials=[melody_draft("
            "foreground_voice='upper', events=[], material_id='x')])",
        ),
        (
            parse_texture_collection,
            "texture_collection(materials=[__import__('os').system('whoami')])",
        ),
        (
            parse_performance_collection,
            "performance_collection(occurrences=[performance_occurrence(node_id='x')])",
        ),
    ),
)
def test_collection_dsl_rejects_unknown_functions_positional_args_and_ids(parse, source) -> None:
    with pytest.raises(WholeScoreCollectionDslError):
        parse(source)


def test_collection_dsl_requires_non_empty_collection() -> None:
    with pytest.raises(WholeScoreCollectionDslError, match="must not be empty"):
        parse_harmonic_collection("harmonic_collection(materials=[])")


def test_harmonic_collection_requires_positive_material_length() -> None:
    source = """harmonic_collection(materials=[
    harmonic_material(
        length_units=0,
        events=[harmonic_event(at_units=0, duration_units=12, root_pitch_class=0, quality='major')],
    ),
])"""

    with pytest.raises(WholeScoreCollectionDslError, match="length_units"):
        parse_harmonic_collection(source)


def test_performance_collection_accepts_only_literal_strings_or_none() -> None:
    source = """performance_collection(occurrences=[
    performance_occurrence(
        timing_profile=None,
        timing_amount=None,
        dynamics_profile=None,
        articulation_profile=None,
        coordination_profile=None,
        pedal_profile=None,
    ),
])"""

    assert parse_performance_collection(source) == (PerformanceOccurrenceDraftV0(),)

    with pytest.raises(WholeScoreCollectionDslError, match="literal"):
        parse_performance_collection(
            source.replace("timing_profile=None", "timing_profile=unknown")
        )
