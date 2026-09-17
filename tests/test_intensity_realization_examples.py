from __future__ import annotations

import json
import statistics

import mido

from llm_musical_composer.intensity_realization import extract_intensity_realization
from llm_musical_composer.intensity_realization_examples import (
    build_realization_pairs,
    expand_composition_for_analysis,
    write_realization_review,
)
from llm_musical_composer.long_form_generation import composition_to_source
from llm_musical_composer.music_dsl import parse_composition


def test_three_families_use_distinct_bases_and_hold_note_count_within_each_pair() -> None:
    pairs = build_realization_pairs()

    assert set(pairs) == {"accent_pulse", "articulation_rest", "voice_coupling_weight"}
    fingerprints = set()
    for _family, levels in pairs.items():
        assert set(levels) == {"low", "high"}
        low = levels["low"]
        high = levels["high"]
        assert low.duration_ms == high.duration_ms == 30_000
        assert low.note_count == high.note_count
        assert [len(item.notes) for item in low.materials] == [
            len(item.notes) for item in high.materials
        ]
        assert parse_composition(composition_to_source(low)) == low
        assert parse_composition(composition_to_source(high)) == high

        base = expand_composition_for_analysis(low)
        fingerprints.add(
            (
                tuple(sorted({note.pitch % 12 for note in base.notes})),
                statistics.median(note.pitch for note in base.notes if note.voice == "lower"),
                tuple(sorted({note.at_ms for note in base.notes}))[:8],
            )
        )
    assert len(fingerprints) == 3


def test_each_family_changes_its_own_mechanism_without_adding_notes() -> None:
    pairs = build_realization_pairs()
    reports = {
        family: {
            level: extract_intensity_realization(expand_composition_for_analysis(composition))
            for level, composition in levels.items()
        }
        for family, levels in pairs.items()
    }

    for levels in reports.values():
        assert levels["low"]["hold"]["note_count"] == levels["high"]["hold"]["note_count"]

    accent = reports["accent_pulse"]
    assert (
        accent["high"]["timing"]["median_voice_local_ioi_ms"]
        < accent["low"]["timing"]["median_voice_local_ioi_ms"]
    )
    assert (
        accent["high"]["accent"]["velocity_p90_minus_median"]
        > accent["low"]["accent"]["velocity_p90_minus_median"]
    )
    assert (
        accent["high"]["voice_coupling"]["synchronous_onset_ratio"]
        == accent["low"]["voice_coupling"]["synchronous_onset_ratio"]
    )
    assert (
        abs(
            accent["high"]["bass"]["low_bass_velocity_median"]
            - accent["low"]["bass"]["low_bass_velocity_median"]
        )
        <= 1
    )
    assert (
        abs(
            accent["high"]["articulation"]["median_duration_to_next_onset_ratio"]
            - accent["low"]["articulation"]["median_duration_to_next_onset_ratio"]
        )
        <= 0.1
    )

    articulation = reports["articulation_rest"]
    assert (
        articulation["high"]["articulation"]["median_duration_to_next_onset_ratio"]
        < articulation["low"]["articulation"]["median_duration_to_next_onset_ratio"]
    )

    coupling = reports["voice_coupling_weight"]
    assert (
        coupling["high"]["voice_coupling"]["synchronous_onset_ratio"]
        > coupling["low"]["voice_coupling"]["synchronous_onset_ratio"]
    )
    assert (
        coupling["high"]["voice_coupling"]["pitch_class_coupling_ratio"]
        > coupling["low"]["voice_coupling"]["pitch_class_coupling_ratio"]
    )
    assert (
        coupling["high"]["bass"]["low_bass_velocity_median"]
        > coupling["low"]["bass"]["low_bass_velocity_median"]
    )


def test_review_export_is_anonymous_and_rereadable(tmp_path) -> None:
    result = write_realization_review(tmp_path)

    manifest = json.loads((tmp_path / "review-manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads((tmp_path / "private-mapping.json").read_text(encoding="utf-8"))
    assert result == manifest
    assert len(manifest["pairs"]) == 3
    assert len(mapping["clips"]) == 6
    assert "family" not in json.dumps(manifest)
    for pair in manifest["pairs"]:
        assert set(pair) == {"pair_id", "clips"}
        assert len(pair["clips"]) == 2
        for name in pair["clips"]:
            midi = mido.MidiFile(tmp_path / name)
            assert midi.length == 30.0
