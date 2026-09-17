"""30 ms発音群に基づく発音頻度候補をScoreSpec上で構成する。"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from llm_musical_composer.control_reference_baseline import normalize_value
from llm_musical_composer.performance_pipeline import (
    HARMONY_INTERVALS,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)

SCHEME_ID = "attack-frequency-v1-structural-grid"
RESOLUTION_FACTOR = 3
EXACT_TOLERANCE = 1e-8
PROTECTED_MATERIALS = frozenset({"ending", "final-tonic", "transition-ab", "transition-ba2"})


@dataclass(frozen=True)
class FrequencyScoreCandidate:
    """発音位置を変更したScoreSpec候補。"""

    direction: str
    stage: int
    score: ScoreSpec
    changed_onsets: dict[str, tuple[int, ...]]
    changed_event_ids: tuple[str, ...]
    added_event_ids: tuple[str, ...]

    @property
    def changed_note_count(self) -> int:
        return len(self.changed_event_ids)

    @property
    def added_note_count(self) -> int:
        return len(self.added_event_ids)


@dataclass(frozen=True)
class MeasuredFrequencyCandidate:
    """RenderedPerformanceまで変換して測った発音頻度候補。"""

    candidate_id: str
    raw_frequency: float
    safe: bool
    changed_note_count: int


@dataclass(frozen=True)
class FrequencyResolution:
    """要求値と、安全候補から選んだ達成値。"""

    requested: float
    target_frequency: float
    candidate_id: str
    achieved_frequency: float
    achieved_normalized: float
    target_error: float
    safe_raw_range: tuple[float, float]
    safe_normalized_range: tuple[float, float]
    status: str


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_endpoints(minimum: object, maximum: object) -> tuple[float, float]:
    prepared_minimum = _finite_number(minimum, "minimum")
    prepared_maximum = _finite_number(maximum, "maximum")
    if prepared_maximum <= prepared_minimum:
        raise ValueError("maximum must be greater than minimum")
    return prepared_minimum, prepared_maximum


def denormalize_frequency(requested: object, *, minimum: object, maximum: object) -> float:
    """`-1.0`から`1.0`の要求をコーパス上の発音群/秒へ戻す。"""

    prepared = _finite_number(requested, "requested")
    if not -1.0 <= prepared <= 1.0:
        raise ValueError("requested frequency must be between -1.0 and 1.0")
    prepared_minimum, prepared_maximum = _validate_endpoints(minimum, maximum)
    return prepared_minimum + ((prepared + 1.0) / 2.0) * (prepared_maximum - prepared_minimum)


def scale_score_resolution(score: ScoreSpec, *, factor: int = RESOLUTION_FACTOR) -> ScoreSpec:
    """音楽上の比率を変えず、ScoreSpecの整数時間解像度を上げる。"""

    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer")
    return replace(
        score,
        score_id=f"{score.score_id}-resolution-x{factor}",
        divisions=score.divisions * factor,
        materials=tuple(
            replace(
                material,
                length_units=material.length_units * factor,
                notes=tuple(
                    replace(
                        note,
                        at_units=note.at_units * factor,
                        duration_units=note.duration_units * factor,
                    )
                    for note in material.notes
                ),
                directions=tuple(
                    replace(direction, at_units=direction.at_units * factor)
                    for direction in material.directions
                ),
                harmonies=tuple(
                    replace(
                        harmony,
                        at_units=harmony.at_units * factor,
                        duration_units=harmony.duration_units * factor,
                    )
                    for harmony in material.harmonies
                ),
            )
            for material in score.materials
        ),
    )


def _protected_onsets(material: ScoreMaterial) -> set[int]:
    onsets = sorted({note.at_units for note in material.notes})
    if not onsets:
        return set()
    if material.material_id in PROTECTED_MATERIALS:
        return set(onsets)
    protected = {onsets[0], onsets[-1]}
    protected.update(note.at_units for note in material.notes if "tenuto" in note.articulations)
    if material.foreground_voice is not None:
        foreground = sorted(
            {note.at_units for note in material.notes if note.voice == material.foreground_voice}
        )
        protected.update(foreground[:4])
    for harmony in material.harmonies:
        end = harmony.at_units + harmony.duration_units
        within = [onset for onset in onsets if harmony.at_units <= onset < end]
        if within:
            protected.add(within[0])
        for voice in ("upper", "lower"):
            voice_onsets = sorted(
                {
                    note.at_units
                    for note in material.notes
                    if note.voice == voice and harmony.at_units <= note.at_units < end
                }
            )
            if voice_onsets:
                protected.add(voice_onsets[0])
    return protected


def _low_sources(material: ScoreMaterial, stage: int) -> list[int]:
    protected = _protected_onsets(material)
    onsets = sorted({note.at_units for note in material.notes})
    accompaniment_only = [
        onset
        for onset in onsets
        if onset not in protected
        and material.foreground_voice is not None
        and all(
            note.voice != material.foreground_voice
            for note in material.notes
            if note.at_units == onset
        )
    ]
    return accompaniment_only[::2] if stage == 1 else accompaniment_only


def _move_target(material: ScoreMaterial, source: int) -> int | None:
    harmony = _harmony_at(material, source)
    if harmony is None:
        return None
    candidates = sorted(
        {note.at_units for note in material.notes if harmony.at_units <= note.at_units < source}
    )
    return candidates[-1] if candidates else None


def _can_move_onset(material: ScoreMaterial, source: int, target: int) -> bool:
    moving = tuple(note for note in material.notes if note.at_units == source)
    fixed = tuple(note for note in material.notes if note.at_units != source)
    return all(
        not any(
            other.voice == note.voice
            and other.pitch == note.pitch
            and _overlaps(other, replace(note, at_units=target))
            for other in fixed
        )
        for note in moving
    )


def build_low_frequency_candidate(score: ScoreSpec, *, stage: int) -> FrequencyScoreCandidate:
    """構造的発音を保護し、発音位置全体を役割順に間引く。"""

    if isinstance(stage, bool) or not isinstance(stage, int) or not 1 <= stage <= 2:
        raise ValueError("stage must be between 1 and 2")
    materials = []
    changed_onsets: dict[str, tuple[int, ...]] = {}
    changed_ids: list[str] = []
    for material in score.materials:
        moves: dict[int, int] = {}
        for source in _low_sources(material, stage):
            target = _move_target(material, source)
            if target is not None and _can_move_onset(material, source, target):
                moves[source] = target
        notes = (
            tuple(
                sorted(
                    (
                        replace(note, at_units=moves[note.at_units])
                        if note.at_units in moves
                        else note
                        for note in material.notes
                    ),
                    key=lambda note: (
                        note.at_units,
                        note.voice,
                        note.pitch,
                        note.event_id,
                    ),
                )
            )
            if moves
            else material.notes
        )
        changed_ids.extend(note.event_id for note in material.notes if note.at_units in moves)
        materials.append(replace(material, notes=notes))
        changed_onsets[material.material_id] = tuple(sorted(moves))
    return FrequencyScoreCandidate(
        "low",
        stage,
        replace(
            score,
            score_id=f"{score.score_id}-frequency-low-{stage}",
            materials=tuple(materials),
        ),
        changed_onsets,
        tuple(sorted(changed_ids)),
        (),
    )


def _harmony_at(material: ScoreMaterial, onset: int):
    matches = tuple(
        harmony
        for harmony in material.harmonies
        if harmony.at_units <= onset < harmony.at_units + harmony.duration_units
    )
    return matches[0] if len(matches) == 1 else None


def _overlaps(left: ScoreNote, right: ScoreNote) -> bool:
    return (
        left.at_units < right.at_units + right.duration_units
        and right.at_units < left.at_units + left.duration_units
    )


def _added_pitch(
    material: ScoreMaterial,
    notes: list[ScoreNote],
    *,
    onset: int,
    sequence_index: int,
) -> tuple[int, str] | None:
    harmony = _harmony_at(material, onset)
    if harmony is None or material.foreground_voice is None:
        return None
    voice = "lower" if material.foreground_voice == "upper" else "upper"
    foreground = [
        note.pitch
        for note in notes
        if note.voice == material.foreground_voice
        and note.at_units <= onset < note.at_units + note.duration_units
    ]
    pitch_classes = {
        (harmony.root_pitch_class + interval) % 12
        for interval in HARMONY_INTERVALS[harmony.quality]
    }
    if voice == "lower":
        ceiling = min(foreground) if foreground else 72
        pitches = [pitch for pitch in range(48, min(ceiling, 72)) if pitch % 12 in pitch_classes]
    else:
        floor = max(foreground) if foreground else 55
        pitches = [pitch for pitch in range(max(48, floor + 1), 85) if pitch % 12 in pitch_classes]

    def valid(pitch: int) -> bool:
        proposed = ScoreNote("candidate", onset, 1, pitch, voice)
        if any(
            note.voice == voice and note.pitch == pitch and _overlaps(note, proposed)
            for note in notes
        ):
            return False
        sounding = sorted(
            {
                note.pitch
                for note in notes
                if note.voice == voice
                and note.at_units <= onset < note.at_units + note.duration_units
            }
            | {pitch}
        )
        return not (len(sounding) >= 2 and sounding[0] < 48 and sounding[1] - sounding[0] < 7)

    available = [pitch for pitch in pitches if valid(pitch)]
    if not available:
        return None
    ordered = sorted(available, key=lambda pitch: (abs(pitch - 56), pitch))
    return ordered[sequence_index % len(ordered)], voice


def _high_positions(material: ScoreMaterial, stage: int) -> list[int]:
    source_units = material.length_units // RESOLUTION_FACTOR
    positions = [
        unit * RESOLUTION_FACTOR + 1 for unit in range(source_units) if stage >= 2 or unit % 2 == 0
    ]
    if stage >= 3:
        positions.extend(
            unit * RESOLUTION_FACTOR + 2 for unit in range(source_units) if unit % 2 == 0
        )
    return sorted(positions)


def build_high_frequency_candidate(score: ScoreSpec, *, stage: int) -> FrequencyScoreCandidate:
    """局所和声内の伴奏パターンを空き3分割位置へ追加する。"""

    if isinstance(stage, bool) or not isinstance(stage, int) or not 1 <= stage <= 3:
        raise ValueError("stage must be between 1 and 3")
    materials = []
    added_ids: list[str] = []
    for material_index, material in enumerate(score.materials):
        if material.material_id in PROTECTED_MATERIALS or not material.harmonies:
            materials.append(material)
            continue
        notes = list(material.notes)
        occupied = {note.at_units for note in notes}
        accompaniment_voice = "lower" if material.foreground_voice == "upper" else "upper"
        final_accompaniment_onset = max(
            note.at_units for note in notes if note.voice == accompaniment_voice
        )
        previous_added_pitch: int | None = None
        for sequence_index, onset in enumerate(_high_positions(material, stage)):
            if onset in occupied or onset > final_accompaniment_onset:
                continue
            selected = _added_pitch(
                material,
                notes,
                onset=onset,
                sequence_index=sequence_index + material_index,
            )
            if selected is None:
                continue
            pitch, voice = selected
            if pitch == previous_added_pitch:
                alternative = _added_pitch(
                    material,
                    notes,
                    onset=onset,
                    sequence_index=sequence_index + material_index + 1,
                )
                if alternative is not None:
                    pitch, voice = alternative
            event_id = f"freq+{material.material_id}+{stage}+{onset}"
            notes.append(ScoreNote(event_id, onset, 1, pitch, voice))
            added_ids.append(event_id)
            occupied.add(onset)
            previous_added_pitch = pitch
        materials.append(
            replace(
                material,
                notes=tuple(
                    sorted(
                        notes,
                        key=lambda note: (
                            note.at_units,
                            note.voice,
                            note.pitch,
                            note.event_id,
                        ),
                    )
                ),
            )
        )
    return FrequencyScoreCandidate(
        "high",
        stage,
        replace(
            score,
            score_id=f"{score.score_id}-frequency-high-{stage}",
            materials=tuple(materials),
        ),
        {},
        tuple(sorted(added_ids)),
        tuple(sorted(added_ids)),
    )


def resolve_measured_frequency(
    requested: object,
    *,
    candidates: tuple[MeasuredFrequencyCandidate, ...],
    minimum: object,
    maximum: object,
) -> FrequencyResolution:
    """安全な実測候補から、要求生値へ最も近い候補を選ぶ。"""

    prepared_minimum, prepared_maximum = _validate_endpoints(minimum, maximum)
    target = denormalize_frequency(requested, minimum=prepared_minimum, maximum=prepared_maximum)
    safe = tuple(
        candidate
        for candidate in candidates
        if candidate.safe and prepared_minimum <= candidate.raw_frequency <= prepared_maximum
    )
    if not safe:
        raise ValueError("no measured frequency candidate is safe inside the corpus range")
    selected = min(
        safe,
        key=lambda candidate: (
            abs(candidate.raw_frequency - target),
            candidate.changed_note_count,
            candidate.candidate_id,
        ),
    )
    safe_raw_range = (
        min(candidate.raw_frequency for candidate in safe),
        max(candidate.raw_frequency for candidate in safe),
    )
    error = abs(selected.raw_frequency - target)
    if error <= EXACT_TOLERANCE:
        status = "exact"
    elif not safe_raw_range[0] <= target <= safe_raw_range[1]:
        status = "unreachable"
    else:
        status = "quantized"
    return FrequencyResolution(
        requested=float(requested),
        target_frequency=target,
        candidate_id=selected.candidate_id,
        achieved_frequency=selected.raw_frequency,
        achieved_normalized=normalize_value(
            selected.raw_frequency,
            minimum=prepared_minimum,
            maximum=prepared_maximum,
        ),
        target_error=error,
        safe_raw_range=safe_raw_range,
        safe_normalized_range=tuple(
            normalize_value(
                value,
                minimum=prepared_minimum,
                maximum=prepared_maximum,
            )
            for value in safe_raw_range
        ),
        status=status,
    )
