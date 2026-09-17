"""音数を固定した`曲の変化`の短い低・中・高対照を作る。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from llm_musical_composer.composition_ir import (
    Composition,
    Material,
    Note,
    Part,
    Pedal,
    Phrase,
    Use,
)
from llm_musical_composer.long_form_generation import composition_to_source
from llm_musical_composer.run_state import atomic_write_bytes
from llm_musical_composer.smf_render import render_composition
from llm_musical_composer.song_change import describe_song_change

MATERIAL_DURATION_MS = 7_500


def _notes(
    prefix: str,
    pitches: Sequence[int],
    onsets: Sequence[int],
    durations: Sequence[int],
    velocities: Sequence[int],
    voice: str,
) -> tuple[Note, ...]:
    if not (len(pitches) == len(onsets) == len(durations) == len(velocities)):
        raise ValueError("note field lengths must match")
    return tuple(
        Note(f"{prefix}-{index}", at, duration, pitch, velocity, voice)
        for index, (pitch, at, duration, velocity) in enumerate(
            zip(pitches, onsets, durations, velocities, strict=True)
        )
    )


def _material(
    material_id: str,
    upper_pitches: Sequence[int],
    lower_pitches: Sequence[int],
    upper_onsets: Sequence[int],
    lower_onsets: Sequence[int],
    *,
    derived_from: str | None = None,
) -> Material:
    prefix = material_id.lower()
    return Material(
        material_id,
        MATERIAL_DURATION_MS,
        (
            *_notes(
                f"{prefix}-u",
                upper_pitches,
                upper_onsets,
                (450,) * len(upper_pitches),
                (62, 66, 60, 68, 63, 65),
                "upper",
            ),
            *_notes(
                f"{prefix}-l",
                lower_pitches,
                lower_onsets,
                (850,) * len(lower_pitches),
                (52, 55, 50, 54),
                "lower",
            ),
        ),
        (
            Pedal(f"{prefix}-pedal-on", 0, 96),
            Pedal(f"{prefix}-pedal-off", 7_250, 0),
        ),
        derived_from,
    )


def _composition(level: str) -> Composition:
    upper_a = (60, 62, 64, 65, 67, 69)
    lower_a = (48, 43, 45, 41)
    upper_a_onsets = (0, 1_000, 2_200, 3_500, 4_800, 6_200)
    lower_a_onsets = (0, 1_800, 3_800, 5_800)
    if level == "low":
        upper_v_onsets = (0, 900, 2_200, 3_600, 4_900, 6_200)
        lower_v_onsets = (0, 1_300, 4_200, 6_200)
        upper_c = (62, 60, 64, 65, 67, 69)
        lower_c = (48, 45, 43, 41)
        upper_c_onsets = (0, 900, 2_200, 3_600, 4_900, 6_200)
        lower_c_onsets = (0, 1_300, 4_200, 6_200)
    elif level == "center":
        upper_v_onsets = (0, 600, 1_800, 3_500, 5_000, 6_200)
        lower_v_onsets = (0, 800, 3_300, 6_200)
        upper_c = (60, 64, 62, 67, 65, 69)
        lower_c = (45, 48, 41, 43)
        upper_c_onsets = (0, 600, 1_800, 3_500, 5_000, 6_200)
        lower_c_onsets = (0, 800, 3_300, 6_200)
    elif level == "high":
        upper_v_onsets = (0, 250, 800, 3_000, 3_900, 6_200)
        lower_v_onsets = (0, 350, 2_700, 6_200)
        upper_c = (60, 69, 62, 67, 64, 65)
        lower_c = (41, 48, 43, 45)
        upper_c_onsets = (0, 250, 800, 3_000, 3_900, 6_200)
        lower_c_onsets = (0, 350, 2_700, 6_200)
    else:
        raise ValueError("level must be low, center, or high")

    materials = (
        _material("A", upper_a, lower_a, upper_a_onsets, lower_a_onsets),
        _material(
            "V",
            upper_a,
            lower_a,
            upper_v_onsets,
            lower_v_onsets,
            derived_from="A",
        ),
        _material("C", upper_c, lower_c, upper_c_onsets, lower_c_onsets),
    )
    return Composition(
        title=f"Song change {level}",
        form=(
            Use("A", "opening", 2),
            Use("V", "contrast", 3),
            Use("C", "climax", 5),
            Use("A", "return", 2),
        ),
        materials=materials,
        parts=(
            Part("P1", "opening", 2, 0, 1),
            Part("P2", "development", 3, 1, 2),
            Part("P3", "climax", 5, 2, 3),
            Part("P4", "return", 2, 3, 4),
        ),
        phrases=(
            Phrase("S1", "P1", "statement", None, 0, 1),
            Phrase("V1", "P2", "variation", "S1", 1, 2, "rhythmic"),
            Phrase("C1", "P3", "contrast", None, 2, 3),
            Phrase("R1", "P4", "return", "S1", 3, 4),
        ),
    )


def build_song_change_triplet() -> dict[str, Composition]:
    return {level: _composition(level) for level in ("low", "center", "high")}


def write_song_change_review(output_dir: Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    triplet = build_song_change_triplet()
    assignments = (("a", "center"), ("b", "high"), ("c", "low"))
    clips: list[str] = []
    private: dict[str, object] = {}
    for letter, level in assignments:
        composition = triplet[level]
        midi_name = f"candidate-{letter}.mid"
        clips.append(midi_name)
        render_composition(composition, output_dir / midi_name)
        atomic_write_bytes(
            output_dir / f"candidate-{letter}.music.py",
            (composition_to_source(composition) + "\n").encode("utf-8"),
        )
        private[midi_name] = {
            "level": level,
            "metrics": describe_song_change(composition),
        }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "question": "3曲を、統一感が強い順から曲中の変化が大きい順へ並べられるか。",
        "clips": clips,
    }
    mapping = {"schema_version": 1, "clips": private}
    atomic_write_bytes(
        output_dir / "review-manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    atomic_write_bytes(
        output_dir / "private-mapping.json",
        (json.dumps(mapping, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return manifest
