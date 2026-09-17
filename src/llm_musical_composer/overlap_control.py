"""コーパス座標の重なり要求を、役割別の音価候補へ解決する。"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace

from llm_musical_composer.control_reference_baseline import normalize_value
from llm_musical_composer.performance_pipeline import (
    HARMONY_INTERVALS,
    PerformanceSpec,
    ScoreMaterial,
    ScoreNote,
    ScoreSpec,
)

SCHEME_ID = "overlap-v1-role-aware-duration"
EXACT_TOLERANCE = 1e-8
PROTECTED_MATERIALS = frozenset({"ending", "final-tonic", "transition-ab", "transition-ba2"})
PARENT_NODES = frozenset({"a1", "b", "a2"})


@dataclass(frozen=True)
class NoteRole:
    """局所和声と声部から決めた音符の役割。"""

    protected: bool
    voice_role: str
    harmony_role: str
    bass_anchor: bool


@dataclass(frozen=True)
class ScoreOverlapCandidate:
    """音価だけを変更したScoreSpec候補。"""

    direction: str
    stage: int
    score: ScoreSpec
    changed_event_ids: tuple[str, ...]
    changed_role_counts: tuple[tuple[str, int], ...]

    @property
    def changed_note_count(self) -> int:
        return len(self.changed_event_ids)


@dataclass(frozen=True)
class CandidateSpec:
    """到達性検査済みの候補構成。"""

    candidate_id: str
    direction: str | None = None
    stage: int | None = None
    articulation_profile: str | None = None


@dataclass(frozen=True)
class MeasuredOverlapCandidate:
    """実演奏へ変換して測った重なり候補。"""

    candidate_id: str
    raw_overlap: float
    safe: bool
    changed_note_count: int
    profile_change_count: int


@dataclass(frozen=True)
class OverlapResolution:
    """要求値と、安全候補から選んだ達成値。"""

    requested: float
    target_overlap: float
    candidate_id: str
    achieved_overlap: float
    achieved_normalized: float
    target_error: float
    safe_raw_range: tuple[float, float]
    safe_normalized_range: tuple[float, float]
    status: str


def _candidate_specs() -> tuple[CandidateSpec, ...]:
    candidates = [CandidateSpec("base")]
    candidates.extend(
        CandidateSpec(f"performance-{profile}", articulation_profile=profile)
        for profile in ("light", "score", "legato")
    )
    candidates.extend(
        CandidateSpec(f"score-short-{stage}", "short", stage) for stage in range(1, 4)
    )
    candidates.extend(
        CandidateSpec(f"score-harmonic-long-{stage}", "harmonic-long", stage)
        for stage in range(1, 5)
    )
    candidates.extend(
        CandidateSpec(f"combined-short-{stage}-light", "short", stage, "light")
        for stage in range(1, 4)
    )
    candidates.extend(
        CandidateSpec(
            f"combined-harmonic-long-{stage}-legato",
            "harmonic-long",
            stage,
            "legato",
        )
        for stage in range(1, 5)
    )
    return tuple(candidates)


CANDIDATE_SPECS = _candidate_specs()


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


def denormalize_overlap(requested: object, *, minimum: object, maximum: object) -> float:
    """`-1.0`から`1.0`の要求をコーパス上の重なり生値へ戻す。"""

    prepared = _finite_number(requested, "requested")
    if not -1.0 <= prepared <= 1.0:
        raise ValueError("requested overlap must be between -1.0 and 1.0")
    prepared_minimum, prepared_maximum = _validate_endpoints(minimum, maximum)
    return prepared_minimum + ((prepared + 1.0) / 2.0) * (prepared_maximum - prepared_minimum)


def _harmony_at(material: ScoreMaterial, note: ScoreNote):
    matches = tuple(
        harmony
        for harmony in material.harmonies
        if harmony.at_units <= note.at_units < harmony.at_units + harmony.duration_units
    )
    return matches[0] if len(matches) == 1 else None


def classify_note(material: ScoreMaterial, note: ScoreNote) -> NoteRole:
    """event_idに依存せず、音符の保護・声部・和声役割を返す。"""

    protected = (
        material.material_id in PROTECTED_MATERIALS
        or "tenuto" in note.articulations
        or note.at_units + note.duration_units == material.length_units
    )
    if material.foreground_voice is None:
        return NoteRole(protected, "unclassified", "unclassified", False)
    voice_role = "foreground" if note.voice == material.foreground_voice else "accompaniment"
    harmony = _harmony_at(material, note)
    if harmony is None:
        harmony_role = "unclassified"
    else:
        interval = (note.pitch - harmony.root_pitch_class) % 12
        if interval == 0:
            harmony_role = "root"
        elif interval == 7:
            harmony_role = "fifth"
        elif interval in HARMONY_INTERVALS[harmony.quality]:
            harmony_role = "chord_color"
        else:
            harmony_role = "nonchord"
    simultaneous_accompaniment = tuple(
        item
        for item in material.notes
        if item.at_units == note.at_units and item.voice != material.foreground_voice
    )
    bass_anchor = bool(
        voice_role == "accompaniment"
        and harmony_role in {"root", "fifth"}
        and simultaneous_accompaniment
        and note.pitch == min(item.pitch for item in simultaneous_accompaniment)
    )
    return NoteRole(protected, voice_role, harmony_role, bass_anchor)


def _short_stage(role: NoteRole) -> int | None:
    if role.protected or role.harmony_role in {"nonchord", "unclassified"} or role.bass_anchor:
        return None
    if role.voice_role == "accompaniment" and role.harmony_role == "chord_color":
        return 1
    if role.voice_role == "accompaniment" and role.harmony_role in {"root", "fifth"}:
        return 2
    if role.voice_role == "foreground":
        return 3
    return None


def _long_stage(role: NoteRole, note: ScoreNote) -> int | None:
    if (
        role.protected
        or role.harmony_role in {"nonchord", "unclassified"}
        or "staccato" in note.articulations
    ):
        return None
    if role.bass_anchor:
        return 1
    if role.voice_role == "accompaniment" and role.harmony_role in {"root", "fifth"}:
        return 2
    if role.voice_role == "foreground":
        return 3
    if role.harmony_role == "chord_color":
        return 4
    return None


def _maximum_harmonic_duration(material: ScoreMaterial, note: ScoreNote, role: NoteRole) -> int:
    bounds = [material.length_units]
    harmony = _harmony_at(material, note)
    if harmony is not None:
        bounds.append(harmony.at_units + harmony.duration_units)
    if role.voice_role == "foreground":
        later_onsets = [
            item.at_units
            for item in material.notes
            if item.voice == note.voice and item.at_units > note.at_units
        ]
    else:
        later_onsets = [
            item.at_units
            for item in material.notes
            if item.voice == note.voice
            and item.pitch == note.pitch
            and item.at_units > note.at_units
        ]
    if later_onsets:
        bounds.append(min(later_onsets))
    return max(1, min(bounds) - note.at_units)


def build_score_candidate(score: ScoreSpec, *, direction: str, stage: int) -> ScoreOverlapCandidate:
    """和声と声部役割の順序に従い、音価だけを変えた候補を作る。"""

    maximum_stage = {"short": 3, "harmonic-long": 4}.get(direction)
    if maximum_stage is None:
        raise ValueError("direction must be short or harmonic-long")
    if isinstance(stage, bool) or not isinstance(stage, int) or not 1 <= stage <= maximum_stage:
        raise ValueError(f"stage must be between 1 and {maximum_stage}")
    changed: list[str] = []
    role_counts: Counter[str] = Counter()
    materials = []
    for material in score.materials:
        notes = []
        for note in material.notes:
            role = classify_note(material, note)
            eligible_stage = _short_stage(role) if direction == "short" else _long_stage(role, note)
            duration = note.duration_units
            if eligible_stage is not None and eligible_stage <= stage:
                if direction == "short":
                    duration = 1
                else:
                    duration = max(
                        duration,
                        _maximum_harmonic_duration(material, note, role),
                    )
            if duration != note.duration_units:
                changed.append(note.event_id)
                role_counts[
                    f"{role.voice_role}:{role.harmony_role}:"
                    f"{'bass' if role.bass_anchor else 'other'}"
                ] += 1
            notes.append(replace(note, duration_units=duration))
        materials.append(replace(material, notes=tuple(notes)))
    return ScoreOverlapCandidate(
        direction=direction,
        stage=stage,
        score=replace(
            score,
            score_id=f"{score.score_id}-overlap-{direction}-{stage}",
            materials=tuple(materials),
        ),
        changed_event_ids=tuple(sorted(changed)),
        changed_role_counts=tuple(sorted(role_counts.items())),
    )


def build_performance_candidate(performance: PerformanceSpec, *, profile: str) -> PerformanceSpec:
    """親区分だけへアーティキュレーション候補を適用する。"""

    if profile not in {"light", "score", "legato"}:
        raise ValueError("profile must be light, score, or legato")
    return replace(
        performance,
        performance_id=f"{performance.performance_id}-overlap-{profile}",
        node_performances=tuple(
            replace(item, articulation_profile=profile) if item.node_id in PARENT_NODES else item
            for item in performance.node_performances
        ),
    )


def resolve_measured_overlap(
    requested: object,
    *,
    candidates: tuple[MeasuredOverlapCandidate, ...],
    minimum: object,
    maximum: object,
) -> OverlapResolution:
    """安全な実測候補から、要求生値へ最も近い候補を決定的に選ぶ。"""

    prepared_minimum, prepared_maximum = _validate_endpoints(minimum, maximum)
    target = denormalize_overlap(requested, minimum=prepared_minimum, maximum=prepared_maximum)
    safe = tuple(
        candidate
        for candidate in candidates
        if candidate.safe and prepared_minimum <= candidate.raw_overlap <= prepared_maximum
    )
    if not safe:
        raise ValueError("no measured overlap candidate is safe inside the corpus range")
    selected = min(
        safe,
        key=lambda candidate: (
            abs(candidate.raw_overlap - target),
            candidate.changed_note_count,
            candidate.profile_change_count,
            candidate.candidate_id,
        ),
    )
    safe_raw_range = (
        min(candidate.raw_overlap for candidate in safe),
        max(candidate.raw_overlap for candidate in safe),
    )
    error = abs(selected.raw_overlap - target)
    if error <= EXACT_TOLERANCE:
        status = "exact"
    elif not safe_raw_range[0] <= target <= safe_raw_range[1]:
        status = "unreachable"
    else:
        status = "quantized"
    return OverlapResolution(
        requested=float(requested),
        target_overlap=target,
        candidate_id=selected.candidate_id,
        achieved_overlap=selected.raw_overlap,
        achieved_normalized=normalize_value(
            selected.raw_overlap,
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
