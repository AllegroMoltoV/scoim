"""音数を固定した「はげしさ」実現方式の短い比較音源を作る。"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from llm_musical_composer.composition_ir import Composition, Material, Note, Pedal, Use
from llm_musical_composer.intensity_realization import extract_intensity_realization
from llm_musical_composer.long_form_generation import composition_to_source
from llm_musical_composer.run_state import atomic_write_bytes
from llm_musical_composer.smf_render import render_composition

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
        Note(f"{prefix}-{index + 1}", at, duration, pitch, velocity, voice)
        for index, (pitch, at, duration, velocity) in enumerate(
            zip(pitches, onsets, durations, velocities, strict=True)
        )
    )


def _pedals(prefix: str, *, articulated: bool = False) -> tuple[Pedal, ...]:
    if articulated:
        return ()
    return (
        Pedal(f"{prefix}-pedal-on-1", 0, 92),
        Pedal(f"{prefix}-pedal-off-1", 1_850, 0),
        Pedal(f"{prefix}-pedal-on-2", 2_050, 92),
        Pedal(f"{prefix}-pedal-off-2", 4_450, 0),
        Pedal(f"{prefix}-pedal-on-3", 4_650, 92),
        Pedal(f"{prefix}-pedal-off-3", 7_350, 0),
    )


def _material(
    family: str,
    level: str,
    section: str,
    upper_pitches: Sequence[int],
    lower_pitches: Sequence[int],
    upper_onsets: Sequence[int],
    lower_onsets: Sequence[int],
    upper_durations: Sequence[int],
    lower_durations: Sequence[int],
    upper_velocities: Sequence[int],
    lower_velocities: Sequence[int],
    *,
    articulated: bool = False,
) -> Material:
    prefix = f"{family}-{level}-{section}"
    return Material(
        material_id=section,
        duration_ms=MATERIAL_DURATION_MS,
        notes=(
            *_notes(
                f"{prefix}-u",
                upper_pitches,
                upper_onsets,
                upper_durations,
                upper_velocities,
                "upper",
            ),
            *_notes(
                f"{prefix}-l",
                lower_pitches,
                lower_onsets,
                lower_durations,
                lower_velocities,
                "lower",
            ),
        ),
        pedals=_pedals(prefix, articulated=articulated),
        derived_from="A" if section == "Aprime" else None,
    )


def _section_pitches(
    upper: Sequence[int], lower: Sequence[int], section: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if section == "A":
        return tuple(upper), tuple(lower)
    if section == "B":
        return tuple(pitch - 2 for pitch in upper), tuple(pitch - 2 for pitch in lower)
    upper_prime = tuple((*upper[2:], *upper[:2]))
    lower_prime = tuple((*lower[1:], lower[0]))
    return upper_prime, lower_prime


def _composition(
    family: str,
    level: str,
    builder: Callable[[str, str], Material],
) -> Composition:
    materials = tuple(builder(section, level) for section in ("A", "B", "Aprime"))
    return Composition(
        title=f"Intensity realization {family} {level}",
        form=(Use("A"), Use("B"), Use("Aprime"), Use("A")),
        materials=materials,
    )


def _accent_pulse(section: str, level: str) -> Material:
    upper, lower = _section_pitches(
        (69, 69, 69, 70, 72, 69, 74, 73),
        (38, 45, 41, 46, 43, 37),
        section,
    )
    if level == "low":
        upper_onsets = (0, 900, 2_100, 3_200, 4_300, 5_450, 6_300, 7_000)
        lower_onsets = (0, 1_500, 2_900, 4_400, 5_700, 6_800)
        upper_velocities = (57, 55, 58, 56, 59, 55, 58, 56)
        lower_velocities = (52, 50, 53, 51, 54, 50)
        upper_durations = (360,) * len(upper)
    else:
        upper_onsets = (0, 450, 900, 2_500, 2_950, 3_400, 5_500, 5_950)
        lower_onsets = (0, 700, 1_800, 4_000, 5_100, 6_800)
        upper_velocities = (82, 48, 48, 86, 50, 50, 90, 52)
        lower_velocities = (82, 48, 48, 86, 52, 52)
        upper_durations = (180,) * len(upper)
    return _material(
        "accent-pulse",
        level,
        section,
        upper,
        lower,
        upper_onsets,
        lower_onsets,
        upper_durations,
        (650,) * len(lower),
        upper_velocities,
        lower_velocities,
    )


def _articulation_rest(section: str, level: str) -> Material:
    upper, lower = _section_pitches(
        (67, 71, 74, 69, 72, 71, 76, 74),
        (43, 50, 47, 52, 45, 50),
        section,
    )
    upper_onsets = (0, 1_000, 1_950, 3_050, 4_200, 5_200, 6_200, 7_000)
    lower_onsets = (0, 1_600, 3_000, 4_500, 5_900, 6_900)
    if level == "low":
        upper_durations = (820, 780, 900, 940, 820, 820, 650, 350)
        lower_durations = (1_300, 1_150, 1_250, 1_150, 800, 400)
    else:
        upper_durations = (220, 180, 240, 200, 260, 180, 220, 180)
        lower_durations = (300, 240, 320, 260, 300, 220)
    return _material(
        "articulation-rest",
        level,
        section,
        upper,
        lower,
        upper_onsets,
        lower_onsets,
        upper_durations,
        lower_durations,
        (61, 63, 60, 64, 62, 65, 63, 61),
        (52, 54, 51, 55, 53, 52),
        articulated=True,
    )


def _voice_coupling_weight(section: str, level: str) -> Material:
    upper, lower = _section_pitches(
        (69, 69, 72, 71, 69, 76, 74, 72),
        (45, 45, 48, 47, 45, 52),
        section,
    )
    upper_onsets = (0, 800, 1_600, 2_600, 3_600, 4_700, 5_800, 6_900)
    if level == "low":
        lower_onsets = (350, 1_600, 2_350, 3_600, 4_550, 6_250)
        lower_durations = (500,) * len(lower)
        lower_velocities = (48, 50, 47, 51, 49, 50)
    else:
        lower_onsets = (0, 800, 1_600, 2_600, 3_600, 4_700)
        lower_durations = (700,) * len(lower)
        lower_velocities = (80, 64, 78, 66, 84, 70)
    return _material(
        "voice-coupling-weight",
        level,
        section,
        upper,
        lower,
        upper_onsets,
        lower_onsets,
        (480,) * len(upper),
        lower_durations,
        (64, 58, 66, 60, 68, 62, 67, 60),
        lower_velocities,
    )


def build_realization_pairs() -> dict[str, dict[str, Composition]]:
    """相互に異なる初期素材から、音数同数の低/高ペアを返す。"""
    builders = {
        "accent_pulse": _accent_pulse,
        "articulation_rest": _articulation_rest,
        "voice_coupling_weight": _voice_coupling_weight,
    }
    return {
        family: {level: _composition(family, level, builder) for level in ("low", "high")}
        for family, builder in builders.items()
    }


def expand_composition_for_analysis(composition: Composition) -> Material:
    """フォームを時間軸へ展開し、素材単体評価器へ渡せる形にする。"""
    offset = 0
    notes: list[Note] = []
    materials = composition.material_by_id
    for use_index, use in enumerate(composition.form):
        material = materials[use.material_id]
        notes.extend(
            replace(note, event_id=f"use-{use_index}-{note.event_id}", at_ms=offset + note.at_ms)
            for note in material.notes
        )
        offset += material.duration_ms
    return Material("expanded", composition.duration_ms, tuple(notes))


def write_realization_review(output_dir: Path) -> dict[str, object]:
    """匿名の3ペアと、開発用の対応表・記法・計測値を書き出す。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = (
        ("pair-1", "accent_pulse", ("low", "high")),
        ("pair-2", "articulation_rest", ("high", "low")),
        ("pair-3", "voice_coupling_weight", ("low", "high")),
    )
    pairs = build_realization_pairs()
    public_pairs: list[dict[str, object]] = []
    private_clips: dict[str, object] = {}
    for pair_id, family, levels in assignments:
        clip_names: list[str] = []
        for letter, level in zip(("a", "b"), levels, strict=True):
            composition = pairs[family][level]
            stem = f"{pair_id}-{letter}"
            midi_name = f"{stem}.mid"
            clip_names.append(midi_name)
            render_composition(composition, output_dir / midi_name)
            atomic_write_bytes(
                output_dir / f"{stem}.music.py",
                (composition_to_source(composition) + "\n").encode("utf-8"),
            )
            private_clips[midi_name] = {
                "family": family,
                "level": level,
                "note_count": composition.note_count,
                "metrics": extract_intensity_realization(
                    expand_composition_for_analysis(composition)
                ),
            }
        public_pairs.append({"pair_id": pair_id, "clips": clip_names})
    manifest: dict[str, object] = {
        "schema_version": 1,
        "question": "各ペアで、どちらがより活発に聞こえるか。音数はペア内で同一。",
        "pairs": public_pairs,
    }
    mapping = {"schema_version": 1, "clips": private_clips}
    atomic_write_bytes(
        output_dir / "review-manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    atomic_write_bytes(
        output_dir / "private-mapping.json",
        (json.dumps(mapping, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return manifest
