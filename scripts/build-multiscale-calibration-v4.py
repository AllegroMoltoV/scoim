"""局所和声から伴奏と前景を組み立て、音域別に検査する3分校正曲を生成する。"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

from llm_musical_composer.performance_pipeline import (
    NodePerformance,
    PerformanceSpec,
    PiecePlan,
    ScoreHarmony,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
    render_performance,
)
from llm_musical_composer.pipeline_dsl import parse_piece_plan, parse_score_spec
from llm_musical_composer.recurrence_quality import (
    analyze_material_harmony,
    analyze_material_vertical_alignment,
    analyze_rendered_harmony,
)

OUTPUT_DIR = Path(".appendix/multiscale-calibration-run-v4")
LEGACY_PATH = Path(__file__).with_name("build-multiscale-calibration-v3.py")
A2_UPPER_ONSETS = (0, 2, 4, 8, 10, 12, 14, 18, 20, 22, 24, 28, 30, 32, 34, 36)
A1_UPPER_ONSETS = (0, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 36)

PROGRESSIONS = {
    "theme": ((9, "minor"), (5, "major"), (0, "major"), (4, "major")),
    "theme-prime": ((9, "minor"), (0, "major"), (5, "major"), (4, "major")),
    "contrast": ((0, "major"), (7, "major"), (5, "major"), (4, "major")),
    "contrast-prime": ((0, "major"), (5, "major"), (7, "major"), (4, "major")),
    "return": ((9, "minor"), (5, "major"), (0, "major"), (4, "major")),
    "return-prime": ((9, "minor"), (5, "major"), (2, "diminished"), (4, "major")),
}


def _load_legacy():
    spec = importlib.util.spec_from_file_location("multiscale_v3", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("v3 generator could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = _load_legacy()
LEGACY_PLAN = LEGACY._plan


def _chord_pitch_classes(root: int, quality: str) -> tuple[int, ...]:
    intervals = {
        "major": (0, 4, 7),
        "minor": (0, 3, 7),
        "diminished": (0, 3, 6),
    }[quality]
    return tuple((root + interval) % 12 for interval in intervals)


def _pitch_for_class(pitch_class: int, low: int, high: int, target: int) -> int:
    candidates = tuple(pitch for pitch in range(low, high + 1) if pitch % 12 == pitch_class)
    return min(candidates, key=lambda pitch: (abs(pitch - target), pitch))


def _harmonies(material_id: str, progression) -> tuple[ScoreHarmony, ...]:
    return tuple(
        ScoreHarmony(f"{material_id}-h{index}", index * 10, 10, root, quality)
        for index, (root, quality) in enumerate(progression)
    )


def _upper_pitches(progression, variant: int) -> tuple[int, ...]:
    pitches: list[int] = []
    patterns = ((0, 1, 2, 1), (1, 2, 0, 2), (2, 1, 0, 1))
    pattern = patterns[variant % len(patterns)]
    for span, (root, quality) in enumerate(progression):
        pitch_classes = _chord_pitch_classes(root, quality)
        tones = tuple(
            sorted(
                _pitch_for_class(pitch_class, 64, 83, 70 + variant + span)
                for pitch_class in pitch_classes
            )
        )
        pitches.extend(tones[index] for index in pattern)
    return tuple(pitches)


def _accompaniment_notes(
    material_id: str,
    progression,
) -> tuple[ScoreNote, ...]:
    notes: list[ScoreNote] = []
    for span, (root, quality) in enumerate(progression):
        start = span * 10
        pitch_classes = _chord_pitch_classes(root, quality)
        bass = _pitch_for_class(root, 40, 52, 46)
        open_pitch = (
            bass + 12
            if quality == "diminished" and bass < 48
            else _pitch_for_class(
                pitch_classes[2],
                bass + (6 if quality == "diminished" else 7),
                68,
                bass + 8,
            )
        )
        middle = _pitch_for_class(pitch_classes[1], 55, 67, 60)
        upper = _pitch_for_class(pitch_classes[2], 52, 67, 57)
        notes.extend(
            (
                ScoreNote(f"{material_id}-l{span}-root", start, 4, bass, "lower"),
                ScoreNote(f"{material_id}-l{span}-open", start, 4, open_pitch, "lower"),
                ScoreNote(f"{material_id}-l{span}-middle", start + 4, 4, middle, "lower"),
                ScoreNote(
                    f"{material_id}-l{span}-upper",
                    start + (6 if span == 3 else 8),
                    4 if span == 3 else 2,
                    upper,
                    "lower",
                ),
            )
        )
    return tuple(notes)


def _a_material(
    name: str,
    *,
    canonical: bool,
    dynamic: str,
    derived_from: str | None,
    variant: int,
) -> ScoreMaterial:
    prefix = "a2" if canonical else "a1"
    material_id = f"{prefix}-{name}"
    progression = PROGRESSIONS[name]
    onsets = A2_UPPER_ONSETS if canonical else A1_UPPER_ONSETS
    melody = _upper_pitches(progression, variant)
    upper: list[ScoreNote] = []
    for index, (onset, pitch) in enumerate(zip(onsets, melody, strict=True)):
        next_onset = onsets[index + 1] if index + 1 < len(onsets) else 40
        upper.append(
            ScoreNote(
                f"{material_id}-u{index}",
                onset,
                min(4 if canonical else 3, next_onset - onset),
                pitch,
                "upper",
                articulations=("accent",) if onset % 10 == 0 else (),
            )
        )
    lower = tuple(
        replace(note, event_id=note.event_id.replace("a2-source", material_id))
        for note in _accompaniment_notes("a2-source", progression)
    )
    return ScoreMaterial(
        material_id,
        40,
        (*upper, *lower),
        derived_from=derived_from,
        directions=LEGACY._directions(material_id, 40, dynamic),
        harmonies=_harmonies(material_id, progression),
        foreground_voice="upper",
    )


def _b_material(
    name: str,
    *,
    dynamic: str,
    derived_from: str | None,
    variant: int,
) -> ScoreMaterial:
    material_id = f"b-{name}"
    progression = (
        ((5, "major"), (0, "major"), (7, "major"), (4, "major"))
        if name.startswith("contrast")
        else ((0, "major"), (5, "major"), (9, "minor"), (4, "major"))
    )
    spans = ((0, 8), (8, 8), (16, 8), (24, 10))
    harmonies = tuple(
        ScoreHarmony(f"{material_id}-h{index}", start, length, root, quality)
        for index, ((start, length), (root, quality)) in enumerate(
            zip(spans, progression, strict=True)
        )
    )
    upper: list[ScoreNote] = []
    for index, ((start, length), (root, quality)) in enumerate(
        zip(spans, progression, strict=True)
    ):
        pitch_classes = _chord_pitch_classes(root, quality)
        root_pitch = _pitch_for_class(pitch_classes[0], 70, 88, 78 + variant)
        third = _pitch_for_class(pitch_classes[1], 72, 90, 81 + variant)
        fifth = _pitch_for_class(pitch_classes[2], 74, 92, 84 + variant)
        upper.extend(
            (
                ScoreNote(f"{material_id}-u{index}-root", start, 3, root_pitch, "upper"),
                ScoreNote(f"{material_id}-u{index}-fifth", start, 3, fifth, "upper"),
                ScoreNote(f"{material_id}-u{index}-third", start + 3, 3, third, "upper"),
                ScoreNote(
                    f"{material_id}-u{index}-tail",
                    start + 6,
                    length - 6,
                    fifth,
                    "upper",
                ),
            )
        )
    lower_onsets = (
        0,
        1,
        3,
        4,
        6,
        7,
        9,
        10,
        12,
        13,
        15,
        16,
        18,
        19,
        21,
        22,
        24,
        25,
        27,
        28,
        29,
        30,
    )
    lower: list[ScoreNote] = []
    for index, onset in enumerate(lower_onsets):
        harmony = next(
            harmony
            for harmony in harmonies
            if harmony.at_units <= onset < harmony.at_units + harmony.duration_units
        )
        pitch_classes = _chord_pitch_classes(harmony.root_pitch_class, harmony.quality)
        pitch_class = pitch_classes[(index + variant) % 3]
        pitch = _pitch_for_class(pitch_class, 52, 71, 62 + variant)
        next_onset = lower_onsets[index + 1] if index + 1 < len(lower_onsets) else 34
        lower.append(
            ScoreNote(
                f"{material_id}-l{index}",
                onset,
                min(2, next_onset - onset),
                pitch,
                "lower",
                articulations=("accent",) if index % 4 == 0 else ("staccato",),
            )
        )
    return ScoreMaterial(
        material_id,
        34,
        (*upper, *lower),
        derived_from=derived_from,
        directions=LEGACY._directions(material_id, 34, dynamic),
        harmonies=harmonies,
        foreground_voice="lower",
    )


def _score() -> ScoreSpec:
    names = tuple(PROGRESSIONS)
    dynamics = ("mp", "mp", "mf", "mf", "mp", "p")
    variants = (0, 0, 1, 1, 0, 0)
    a2_sources = (None, "a2-theme", None, "a2-contrast", "a2-theme", "a2-theme-prime")
    aligned = tuple(
        _a_material(
            name,
            canonical=True,
            dynamic=dynamics[index],
            derived_from=a2_sources[index],
            variant=variants[index],
        )
        for index, name in enumerate(names)
    )
    broken = tuple(
        _a_material(
            name,
            canonical=False,
            dynamic=dynamics[index],
            derived_from=f"a2-{name}",
            variant=variants[index],
        )
        for index, name in enumerate(names)
    )
    b_sources = (None, "b-theme", None, "b-contrast", "b-theme", "b-theme-prime")
    b_materials = tuple(
        _b_material(
            name,
            dynamic=dynamics[index],
            derived_from=b_sources[index],
            variant=variants[index],
        )
        for index, name in enumerate(names)
    )
    return ScoreSpec(
        "multiscale-score-v4",
        4,
        (*aligned, *broken, *b_materials, LEGACY._ending()),
    )


def _plan() -> PiecePlan:
    plan = LEGACY_PLAN()
    return replace(
        plan,
        plan_id="multiscale-calibration-v4",
        title="局所和声から弾き直す",
        nodes=tuple(replace(node, harmonic_focus=None) for node in plan.nodes),
    )


def _performance() -> PerformanceSpec:
    items = [
        NodePerformance(
            "a1",
            timing_profile="savor",
            timing_amount="moderate",
            dynamics_profile="shape",
            articulation_profile="legato",
            coordination_profile="score",
        ),
        NodePerformance(
            "b",
            timing_profile="build",
            timing_amount="moderate",
            dynamics_profile="build",
            articulation_profile="score",
            coordination_profile="score",
        ),
        NodePerformance(
            "a2",
            timing_profile="flow",
            timing_amount="moderate",
            dynamics_profile="release",
            articulation_profile="light",
            coordination_profile="aligned",
        ),
    ]
    phrase_timings = {
        "a1": ("savor", "neutral", "release"),
        "b": ("build", "build", "release"),
        "a2": ("flow", "build", "release"),
    }
    for section in ("a1", "b", "a2"):
        for index, timing in enumerate(phrase_timings[section]):
            items.append(
                NodePerformance(
                    f"{section}-{index}",
                    timing_profile=timing,
                    timing_amount="subtle",
                    pedal_profile="harmony_legato",
                )
            )
    items.append(NodePerformance("a2-2-2", pedal_profile="clear"))
    return PerformanceSpec(
        "multiscale-performance-v4",
        180_000,
        62,
        "narrative-v1",
        tuple(items),
    )


def main(maximum_target_ratio: float = 0.85) -> int:
    LEGACY.OUTPUT_DIR = OUTPUT_DIR
    LEGACY._score = _score
    LEGACY._plan = _plan
    LEGACY._performance = _performance
    result = LEGACY.main(maximum_target_ratio=maximum_target_ratio)

    plan = parse_piece_plan((OUTPUT_DIR / "inputs/piece-plan.music.py").read_text(encoding="utf-8"))
    score = parse_score_spec((OUTPUT_DIR / "inputs/score.music.py").read_text(encoding="utf-8"))
    rendered = render_performance(plan, score, _performance())
    harmonic = tuple(
        analyze_material_harmony(
            score,
            material.material_id,
            low_pitch_boundary=48,
            minimum_low_spacing_semitones=7,
        )
        for material in score.materials
        if material.harmonies
    )
    rendered_harmony = analyze_rendered_harmony(plan, score, rendered)
    a1_alignment = tuple(
        analyze_material_vertical_alignment(material, minimum_ratio=0.40)
        for material in score.materials
        if material.material_id.startswith("a1-")
    )
    a2_alignment = tuple(
        analyze_material_vertical_alignment(material, minimum_ratio=0.70)
        for material in score.materials
        if material.material_id.startswith("a2-")
    )
    failures = [
        f"harmony:{item.material_id}"
        for item in harmonic
        if not item.passes or item.accompaniment_chord_attacks < 1 or item.arpeggio_candidates < 1
    ]
    failures.extend(
        f"score-alignment:{item.material_id}"
        for item in (*a1_alignment, *a2_alignment)
        if not item.passes
    )
    if not rendered_harmony.passes:
        failures.append("rendered-harmony")
    if failures:
        raise ValueError("v4 harmony gates failed: " + ", ".join(failures))

    result_path = OUTPUT_DIR / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["harmony"] = {
        "materials": [
            {
                "material_id": item.material_id,
                "foreground_voice": next(
                    material.foreground_voice
                    for material in score.materials
                    if material.material_id == item.material_id
                ),
                "accompaniment_voice": item.accompaniment_voice,
                "solo_attacks": item.accompaniment_solo_attacks,
                "chord_attacks": item.accompaniment_chord_attacks,
                "arpeggio_candidates": item.arpeggio_candidates,
                "chord_tone_ratio": item.accompaniment_chord_tone_ratio,
                "low_spacing_violations": item.low_spacing_violations,
                "unresolved_foreground_non_chord_tones": (
                    item.unresolved_foreground_non_chord_tones
                ),
            }
            for item in harmonic
        ],
        "pedal_carryover_violations": rendered_harmony.pedal_carryover_violations,
        "low_pitch_boundary": 48,
        "minimum_low_spacing_semitones": 7,
    }
    payload["a1_material_alignments"] = [
        {
            "material_id": item.material_id,
            "shared_attack_ratio": item.shared_attack_ratio,
        }
        for item in a1_alignment
    ]
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(result_path.resolve())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
